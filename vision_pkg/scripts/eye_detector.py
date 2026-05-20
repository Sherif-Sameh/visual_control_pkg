#!/usr/bin/env python3

"""
ROS node for performing eye pose detection using differentiable rendering (Kaolin)
"""

import logging
import math
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from isaac_ros_apriltag_interfaces.msg import AprilTagDetection, AprilTagDetectionArray
from kaolin.render.camera import Camera
from numpy.typing import NDArray
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty, Header
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from torch import LongTensor, Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.io import read_image

import vc_core.dr.common as common
import vc_core.dr.kaolin as vc_kal
import vc_core.utils.geometry.pose as pose_utils
from vc_core.segmentation.sam import SAM2LiveVideo, SAMPromptConfig
from vc_core.utils.ros.tf2 import lookup_transform

logging.getLogger("kaolin.rep.surface_mesh").setLevel(logging.ERROR)


class EyeDetector(Node):
    """ROS node for performing eye pose detection using differentiable rendering (Kaolin)."""

    EYE_ROT_OFFSET = torch.tensor(R.from_quat([1.0, 0.0, 0.0, 0.0]).as_matrix()).float().cuda()

    def __init__(self):
        super().__init__("eye_detector")

        # Declare ROS parameters
        self.declare_parameter("frame.marker", rclpy.Parameter.Type.STRING)
        self.declare_parameter("frame.eye_gt", rclpy.Parameter.Type.STRING)
        self.declare_parameter("ref.dist", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("ref.tol", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("ref.pose", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("sam.variant", rclpy.Parameter.Type.STRING)
        self.declare_parameter("sam.prompt.n_pos", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.mesh.path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("dr.mesh.offsets", rclpy.Parameter.Type.STRING)
        self.declare_parameter("dr.mesh.texture", rclpy.Parameter.Type.STRING)
        self.declare_parameter("dr.mesh.scale", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.mesh.elev_lim", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("dr.mesh.azim_lim", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("dr.model.n_rep", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.model.pos", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("dr.model.tan", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("dr.model.scale", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.shader.sigma", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.shader.boxlen", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.shader.knum", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.raster.size", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.raster.backend", rclpy.Parameter.Type.STRING)
        self.declare_parameter("dr.optim.loss", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("dr.optim.loss.weights", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("dr.optim.loss.tan_norm", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.optim.lr", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.optim.sched.eta_min", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.optim.n_iter.init", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.optim.n_iter", rclpy.Parameter.Type.INTEGER)

        # Initialize non-ROS class attributes
        ref_pose = self.get_parameter("ref.pose").value
        if not torch.cuda.is_available():
            self.get_logger().warn("CUDA is not available. Kaolin cannot run on CPU.")
            rclpy.try_shutdown()
        self._device = torch.device("cuda")
        self._frame_mk = self.get_parameter("frame.marker").value
        self._frame_eye_gt = self.get_parameter("frame.eye_gt").value
        self._ref_dist = self.get_parameter("ref.dist").value
        self._ref_tol = self.get_parameter("ref.tol").value
        self._ref_pose = (np.array(ref_pose[:3]), R.from_quat(ref_pose[3:], scalar_first=True))
        self._n_rep = self.get_parameter("dr.model.n_rep").value
        self._size = self.get_parameter("dr.raster.size").value
        self._bridge = CvBridge()
        self._reset()
        self._cameras, self._cam_K, self._sam = None, None, None
        self._optim = self._init_optim()

        # Initialize ROS attributes
        self._timer = self.create_timer(0.05, self.callback_timer)
        self._pub_pose = self.create_publisher(AprilTagDetectionArray, "/eye_detector/pose", 1)
        self._pub_perr = self.create_publisher(PoseStamped, "/eye_detector/pose_error", 10)
        self._pub_seg = self.create_publisher(Image, "/eye_detector/segmentation", 1)
        self._sub_img = self.create_subscription(Image, "/image", self.callback_img, 0)
        self._sub_cam_info = self.create_subscription(
            CameraInfo, "/camera_info", self.callback_cam_info, 1
        )
        self._sub_cam_twist = self.create_subscription(
            TwistStamped, "/camera_twist", self.callback_cam_twist, 1
        )
        self._sub_rst = self.create_subscription(
            Empty, "/eye_detector/restart", self.callback_rst, 1
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.add_on_set_parameters_callback(self.callback_params)

    def publish_pose(self, tvec: NDArray, rmat: NDArray) -> None:
        """Publish the estimated eye wrt camera pose."""
        quat = R.from_matrix(rmat).as_quat()
        msg = AprilTagDetectionArray()
        msg.header = self._cam_header
        dtn = AprilTagDetection()
        dtn.id = 0
        dtn.pose.pose.pose.position.x = float(tvec[0])
        dtn.pose.pose.pose.position.y = float(tvec[1])
        dtn.pose.pose.pose.position.z = float(tvec[2])
        dtn.pose.pose.pose.orientation.x = float(quat[0])
        dtn.pose.pose.pose.orientation.y = float(quat[1])
        dtn.pose.pose.pose.orientation.z = float(quat[2])
        dtn.pose.pose.pose.orientation.w = float(quat[3])
        msg.detections.append(dtn)
        self._pub_pose.publish(msg)

    def publish_tf(self, tvec: NDArray, rmat: NDArray) -> None:
        """Publish the estimated eye wrt camera transform to the TF tree."""
        quat = R.from_matrix(rmat).as_quat()
        transform = TransformStamped()
        transform.header = self._cam_header
        transform.child_frame_id = "eye_est"
        transform.transform.translation.x = float(tvec[0])
        transform.transform.translation.y = float(tvec[1])
        transform.transform.translation.z = float(tvec[2])
        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])
        self._tf_broadcaster.sendTransform(transform)

    def publish_perr(self, transform_gt: TransformStamped | None) -> None:
        """Publish the estimated eye wrt camera pose error."""
        if transform_gt is None:
            return
        # extract ground truth and estimated poses
        tvec_gt, rot_gt = pose_utils.from_transform_gm(transform_gt.transform)
        tvec, rot = self._pose[0].cpu().numpy(), R.from_matrix(self._pose[1].cpu().numpy())
        # compute pose error
        tvec_err = tvec - tvec_gt
        quat_err = (rot * rot_gt.inv()).as_quat()
        # publish pose error message
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(tvec_err[0])
        msg.pose.position.y = float(tvec_err[1])
        msg.pose.position.z = float(tvec_err[2])
        msg.pose.orientation.x = float(quat_err[0])
        msg.pose.orientation.y = float(quat_err[1])
        msg.pose.orientation.z = float(quat_err[2])
        msg.pose.orientation.w = float(quat_err[3])
        self._pub_perr.publish(msg)

    def publish_seg(self, img: Tensor, mask: Tensor) -> None:
        """Publish alpha-blended RGB + segmentation mask image."""
        alpha = 0.6
        color = torch.tensor((30.0, 144.0, 255.0)).view(1, 1, 3).div(255).to(self._device)
        mask = mask.unsqueeze(-1)
        seg_img = torch.where(mask, img * (1 - alpha) + mask * color * alpha, img)
        seg_img = seg_img.mul(255).byte().cpu().numpy()
        msg = self._bridge.cv2_to_imgmsg(
            seg_img, encoding="rgb8", header=Header(stamp=self.get_clock().now().to_msg())
        )
        self._pub_seg.publish(msg)

    def callback_timer(self) -> None:
        if self._pose is None and self._cam_header is None:
            return
        # get eye wrt camera pose directly or through marker pose
        if self._pose is None:
            # extract marker pose
            transform = lookup_transform(self._cam_header.frame_id, self._frame_mk, self._tf_buffer)
            if transform is None:
                return
            tvec, rot = pose_utils.from_transform_gm(transform.transform)
            # compose with reference eye wrt marker pose
            tvec, rot = pose_utils.pose_mult((tvec, rot), self._ref_pose)
            rmat = rot.as_matrix()
            # init pose if stationary and camera to eye distance < reference distance
            if self._is_stationary and np.linalg.norm(tvec) < self._ref_dist:
                self._pose = (
                    torch.tensor(tvec).to(dtype=torch.float32, device=self._device),
                    torch.tensor(rmat).to(dtype=torch.float32, device=self._device),
                )
            # publish indirect eye pose
            self.publish_pose(tvec, rmat)
        else:
            # extract pose directly
            tvec = self._pose[0].cpu().numpy()
            rmat = self._pose[1].cpu().numpy()
        # send tf transform
        self.publish_tf(tvec, rmat)

    def callback_img(self, msg: Image) -> None:
        self._cam_header = msg.header
        if self._pose is None or self._cameras is None:
            return
        # store GT transform if available
        transform_gt = None
        if self._frame_eye_gt:
            transform_gt = lookup_transform(
                msg.header.frame_id, self._frame_eye_gt, self._tf_buffer
            )
        # load image as tensor
        img = torch.frombuffer(msg.data, dtype=torch.uint8).view(msg.height, msg.width, 3)
        img = img.to(dtype=torch.float32, device=self._device).div(255)
        # initialize segmentation and pose models
        not_init = False
        if self._center is None:
            not_init = True
            self._init_seg_mask(img)
            tvec, rmat = self._apply_rot_offset(self._pose[0], self._pose[1])
            self._optim.resample_model_params(tvec, rmat)
        # run segmentation model and pose optimizer
        img = img[:, self._w_h_2 : self._w_h_2 + msg.height]
        mask = self._sam.segment(img.permute(2, 0, 1), [])
        self._pose = self._optimize(img, mask, msg.height)
        if not_init:
            self._n_iter = self.get_parameter("dr.optim.n_iter").value
            self._optim = self._init_optim(optimizer=True)
        # publish everything and prepare for next callback
        self.publish_pose(self._pose[0].cpu().numpy(), self._pose[1].cpu().numpy())
        self.publish_perr(transform_gt)
        self.publish_seg(img, mask)

    def callback_cam_info(self, msg: CameraInfo) -> None:
        if self._cameras is None:
            self._w_h_2 = (msg.width - msg.height) // 2
            self._cam_K = torch.frombuffer(msg.k, dtype=torch.float64)
            self._cam_K = self._cam_K.view(3, 3).to(dtype=torch.float32, device=self._device)
            self._cam_K[0, 2] *= -1
            camera = Camera.from_args(
                view_matrix=torch.eye(4),
                focal_x=msg.k[0],
                x0=msg.k[2] - msg.width / 2,
                y0=msg.k[5] - msg.height / 2,
                height=self._size,
                width=self._size,
                dtype=torch.float32,
            )
            self._cameras = Camera.cat([camera] * self._n_rep).to(self._device)
            self._sam = self._init_seg(msg.height)

    def callback_cam_twist(self, msg: TwistStamped) -> None:
        lin, ang = msg.twist.linear, msg.twist.angular
        v_norm = math.sqrt(lin.x * lin.x + lin.y * lin.y + lin.z * lin.z)
        w_norm = math.sqrt(ang.x * ang.x + ang.y * ang.y + ang.z * ang.z)
        self._is_stationary = v_norm < self._ref_tol[0] and w_norm < self._ref_tol[1]

    def callback_rst(self, msg: Empty) -> None:
        self._reset()

    def callback_params(self, params: list[rclpy.Parameter]) -> SetParametersResult:
        result = SetParametersResult()
        result.successful = True
        result.reason = "success"

        def failed(reason: str) -> None:
            result.successful = False
            result.reason = reason

        # Check for params that are allowed to change at runtime
        ParamType = rclpy.Parameter.Type
        for param in params:
            if param.name in [
                "dr.shader.sigma",
                "dr.shader.boxlen",
                "dr.optim.lr",
                "dr.optim.sched.eta_min",
            ] and (param.type_ != ParamType.DOUBLE or param.value <= 0):
                failed(f"{param.name} must be double > 0.")
                break
            if param.name == "dr.optim.loss" and param.type_ != ParamType.STRING_ARRAY:
                failed("dr.optim.loss must be string array.")
                break
        if result.successful:
            self._reset()
            self._optim = self._init_optim(model=True, renderer=True, optimizer=True)
        return result

    def _init_seg(self, size: int) -> SAM2LiveVideo:
        """Initialize the segmentation model."""
        assert size % 32 == 0, "Image size must be a multiple of 32 for SAM2."
        variant = self.get_parameter("sam.variant").value
        prompt_n_pos = self.get_parameter("sam.prompt.n_pos").value
        return SAM2LiveVideo(
            var=variant,
            cfg=SAMPromptConfig(n_pos=prompt_n_pos, n_neg=0),
            device=self._device,
            max_obj_num=1,
            overrides={"imgsz": size},
        )

    def _init_seg_mask(self, img: Tensor) -> None:
        """First-time initialization of segmentation mask and centroid."""
        # sample prompts and segment full image
        H = img.shape[0]
        img_sam = img[:, self._w_h_2 : self._w_h_2 + H].permute(2, 0, 1)
        self._sam.point_prompts, self._sam.label_prompts = self._init_seg_prompts()
        if not self._sam.point_prompts:
            self._sam.sample_points(img_sam)
            self._store_prompts((self._sam.point_prompts, self._sam.label_prompts))
        mask = self._sam.segment(img_sam, [0] * len(self._sam.point_prompts), update_memory=True)
        # compute mask centroid for crops
        self._center = self._centroid(mask).long()
        _, self._center = self._crop_img(
            img[:, self._w_h_2 : self._w_h_2 + H], self._center, self._size
        )

    def _init_seg_prompts(self) -> tuple[list[tuple[int, int]], list[int]]:
        """Initialize segmentation prompts from pickle file if it exists."""
        path = Path(__file__).parent / "prompts/prompts.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return ([], [])

    def _store_prompts(self, prompts: tuple[list[tuple[int, int]], list[int]]) -> None:
        """Store the collected segmentation prompts."""
        path = Path(__file__).parent / "prompts/prompts.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(prompts, f)

    def _init_optim(self, **kwargs) -> vc_kal.optim.EyePoseOptimizer:
        """Initialize the DR-based eye pose optimizer."""
        has_optim = hasattr(self, "_optim")
        mesh = (
            self._init_optim_mesh()
            if not has_optim or kwargs.get("mesh", False)
            else self._optim._mesh
        )
        model = (
            self._init_optim_model()
            if not has_optim or kwargs.get("model", False)
            else self._optim._model
        )
        mesh_renderer = (
            self._init_optim_renderer()
            if not has_optim or kwargs.get("renderer", False)
            else self._optim._renderer
        )
        if not has_optim or kwargs.get("optimizer", False):
            optim_params = {
                k: v.value for k, v in self.get_parameters_by_prefix("dr.optim").items()
            }
            loss_fn = self._init_optim_loss_fn(optim_params)
            tan_norm_w = optim_params["loss.tan_norm"]
            lr, lr_sched_cfg = self._init_optim_lr_and_sched(optim_params)
        else:
            loss_fn = self._optim._loss_fn
            tan_norm_w = self._optim._tan_norm_w
            lr, lr_sched_cfg = self._optim._lr, self._optim._sched_cfg
        return vc_kal.optim.EyePoseOptimizer(
            mesh,
            model,
            mesh_renderer,
            loss_fn,
            tan_norm_w=tan_norm_w,
            lr=lr,
            lr_sched_cfg=lr_sched_cfg,
        )

    def _init_optim_mesh(self) -> vc_kal.mesh.EyeObjMesh:
        """Intialize the obj mesh used by the optimizer."""
        # load mesh from obj
        mesh_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.mesh").items()}
        mesh = vc_kal.mesh.EyeObjMesh(
            mesh_params["path"],
            n_rep=self._n_rep,
            elev_lim=mesh_params["elev_lim"],
            azim_lim=mesh_params["azim_lim"],
        )
        mesh.mesh.vertices *= mesh_params["scale"]
        self._mesh_offset = torch.tensor(
            [0.0, 0.0, mesh.mesh.vertices[..., -1].amax()], device=self._device
        )
        # ensure that face_features is not None
        n_face = mesh.mesh.faces.shape[0]
        mesh.mesh.face_features = torch.zeros([self._n_rep, n_face, 3, 0])
        # update mesh vertices with calibrated offsets
        if len(mesh_params["offsets"]) > 0:
            offsets = torch.load(mesh_params["offsets"], weights_only=False)
            mesh.mesh.vertices += offsets
            self.get_logger().info(f"Mesh using vertex offsets from {mesh_params['offsets']}.")
        # update UVs and texture to Kaolin conventions
        mesh.mesh.face_uvs = 1 - mesh.mesh.face_uvs
        if len(mesh_params["texture"]) > 0:
            texture = read_image(mesh_params["texture"])
            self.get_logger().info(f"Mesh using texture from {mesh_params['texture']}.")
        else:
            texture = mesh.mesh.materials[0][0]["map_Kd"].permute(2, 0, 1).contiguous()
            self.get_logger().info("Mesh using default texture.")
        mesh.mesh.materials[0][0]["map_Kd"] = texture.float().div(255).cuda()
        return mesh.to(device=self._device)

    def _init_optim_model(self) -> common.model.EyePoseModel:
        """Initialize the eye pose model used by the optimizer."""
        model_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.model").items()}
        model = common.model.EyePoseModel(
            torch.rand(3),
            torch.eye(3),
            torch.tensor(model_params["pos"]),
            torch.tensor(model_params["tan"]),
            n_rep=model_params["n_rep"],
            scale=model_params["scale"],
        )
        return torch.compile(model.to(device=self._device))

    def _init_optim_renderer(self) -> vc_kal.render.MeshRenderer:
        """Initialize the renderer used by the optimizer."""
        shader_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.shader").items()}
        raster_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.raster").items()}
        shader_params["sigmainv"] = 1 / shader_params["sigma"]
        shader_params.pop("sigma")
        blend_params = vc_kal.render.BlendParams(**shader_params)
        raster_settings = vc_kal.render.RasterizationSettings(
            image_size=raster_params["size"], backend=raster_params["backend"]
        )
        return vc_kal.render.MeshRenderer(
            rasterizer=vc_kal.render.MeshRasterizer(raster_settings=raster_settings),
            shader=vc_kal.render.ComposeShader(
                [
                    vc_kal.render.SoftSilhouetteShader(blend_params=blend_params),
                    vc_kal.render.HardColorAmbientShader(raw_texture=False, uvs_origin="Kaolin"),
                ]
            ),
        ).to(device=self._device)

    def _init_optim_loss_fn(
        self, optim_params: dict[str, Any]
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Initialize the loss function used by the optimizer."""
        losses = [optim_params["loss"][0], "masked_loss"]
        return torch.compile(
            common.losses.build_combined_loss_fn(
                losses,
                [slice(1), slice(4)],
                weights=optim_params["loss.weights"],
                device=self._device,
                reduction="mean",
                dim=(1, 2, 3),
                kwargs=[{}, {"inner_fn_name": optim_params["loss"][1]}],
            )
        )

    def _init_optim_lr_and_sched(self, optim_params: dict[str, Any]):
        """Initialize the learning rate and LR scheduler config used by the optimzier."""
        lr = optim_params["lr"]
        lr_sched_cfg = vc_kal.optim.EyePoseOptimizer.LRSchedulerCfg(
            cls=CosineAnnealingLR,
            kwargs={"T_max": self._n_iter, "eta_min": optim_params["sched.eta_min"]},
        )
        return lr, lr_sched_cfg

    def _optimize(self, img: Tensor, mask: Tensor, height: int) -> tuple[Tensor, Tensor]:
        """Run optimization for eye pose parameters."""
        uv_centroid = self._centroid(mask).flip(0).repeat(self._n_rep, 1)
        uv_centroid[:, 0] = -(uv_centroid[:, 0] + self._w_h_2)
        self._optim.init_from_pinhole_model(uv_centroid, self._cam_K)
        self._cameras.intrinsics.x0 -= self._center[1] - height // 2
        self._cameras.intrinsics.y0 += self._center[0] - height // 2
        target = self._get_target(img, mask)
        tvec, rmat = self._optim.optimize(target, n_iter=self._n_iter, cameras=self._cameras)
        self._cameras.intrinsics.x0 += self._center[1] - height // 2
        self._cameras.intrinsics.y0 -= self._center[0] - height // 2
        tvec = tvec + rmat @ self._mesh_offset
        return self._apply_rot_offset(tvec, rmat)

    def _get_target(self, img: Tensor, mask: Tensor) -> Tensor:
        """Get the target image from stored silhouette and RGB images."""
        silhouette = mask.unsqueeze(-1)
        target = torch.cat([silhouette, img * silhouette], dim=-1)
        target, _ = self._crop_img(target, self._center, self._size)
        return target.unsqueeze(0).expand(self._n_rep, -1, -1, -1)

    @staticmethod
    def _centroid(mask: Tensor) -> Tensor:
        """Compute centroid of 2D mask."""
        return mask.nonzero().float().mean(dim=0)

    @staticmethod
    def _crop_img(img: Tensor, center: LongTensor, size: int) -> tuple[Tensor, Tensor]:
        """Crop input image at given center to prepare for rendering."""
        H, W = img.shape[-3:-1]
        top = max(int(center[0] - size / 2), 0)
        left = max(int(center[1] - size / 2), 0)
        right = min(left + size, W)
        bottom = min(top + size, H)
        center[0] = (top + bottom) // 2
        center[1] = (left + right) // 2
        return img[..., top:bottom, left:right, :], center

    def _apply_rot_offset(self, tvec: Tensor, rmat: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the fixed rotation offset between eye wrt camera poses.

        This offset is caused by two rotations. Firstly between the OpenGL and ROS camera
        conventions. Secondly between the eye mesh's coordinate frame and the eye frame we use.
        Both are corrected by a rotation of 180deg around the X-axis.
        """
        tvec_new = self.EYE_ROT_OFFSET @ tvec
        rot_new = self.EYE_ROT_OFFSET @ rmat @ self.EYE_ROT_OFFSET
        return tvec_new, rot_new

    def _reset(self) -> None:
        """Reset all attributes that are initialized from callbacks."""
        self._cam_header = None
        self._center = None
        self._pose = None
        self._is_stationary = False
        self._n_iter = self.get_parameter("dr.optim.n_iter.init").value


def main(args=None):
    rclpy.init(args=args)
    eye_detector = EyeDetector()
    rclpy.spin(eye_detector)
    eye_detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
