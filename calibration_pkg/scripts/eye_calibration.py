#!/usr/bin/env python3

"""
ROS node for performing eye texture calibration using differentiable rendering (Kaolin)
"""

from typing import Any, Callable

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Transform, Twist
from kaolin.render.camera import Camera
from numpy.typing import NDArray
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty, Header
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from torch import LongTensor, Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

import vc_core.dr.common as common
import vc_core.dr.kaolin as vc_kal
from vc_core.segmentation.sam import SAM2, SAMPromptConfig
from vc_core.utils.ros.tf2 import lookup_transform


class EyeCalibration(Node):
    """ROS node for performing eye texture calibration using differentiable rendering (Kaolin)."""

    def __init__(self):
        super().__init__("eye_calibration")

        # Declare ROS parameters
        self.declare_parameter("output_path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("frame.cam", rclpy.Parameter.Type.STRING)
        self.declare_parameter("frame.marker", rclpy.Parameter.Type.STRING)
        self.declare_parameter("frame.eye_gt", rclpy.Parameter.Type.STRING)
        self.declare_parameter("ref.marker_id", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("ref.pose", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("ref.pose.tol", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("sam.variant", rclpy.Parameter.Type.STRING)
        self.declare_parameter("sam.prompt.n_pos", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("sam.prompt.n_neg", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("pose.n_view_sqrt", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("pose.dist", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("pose.range.elev", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("pose.range.azim", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.mesh.path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("dr.model.type", rclpy.Parameter.Type.STRING)
        self.declare_parameter("dr.model.res", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.model.scale", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.model.n_level", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.model.log2_hsz", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.model.mlp_n_layer", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.shader.sigma", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.shader.boxlen", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.shader.knum", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.raster.size", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.raster.backend", rclpy.Parameter.Type.STRING)
        self.declare_parameter("dr.optim.loss", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("dr.optim.loss.weights", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("dr.optim.loss.symmetry", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.optim.loss.tan_norm", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.optim.lr", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.optim.sched.T_0", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.optim.sched.T_mult", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.optim.sched.eta_min", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.optim.n_iter.init", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.optim.n_iter.text", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.optim.n_iter.full", rclpy.Parameter.Type.INTEGER)

        # Initialize non-ROS class attributes
        if not torch.cuda.is_available():
            self.get_logger().warn("CUDA is not available. Kaolin cannot run on CPU.")
            rclpy.try_shutdown()
        self._device = torch.device("cuda")
        self._frames = {k: v.value for k, v in self.get_parameters_by_prefix("frame").items()}
        self._ref_id = self.get_parameter("ref.marker_id").value
        self._ref_pose = np.array(self.get_parameter("ref.pose").value)
        self._ref_ptol = np.array(self.get_parameter("ref.pose.tol").value)
        self._n_view = self.get_parameter("pose.n_view_sqrt").value ** 2
        self._size = self.get_parameter("dr.raster.size").value
        self._bridge = CvBridge()
        self._sam = self._init_seg()
        self._poses = self._init_poses()
        self._optim = self._init_optim()
        self._reset()

        # Initialize ROS attributes
        self._timer = self.create_timer(0.5, self.callback_timer)
        self._pub_target = self.create_publisher(
            MultiDOFJointTrajectory, "/eye_calibration/command", 0
        )
        self._pub_perr = self.create_publisher(PoseStamped, "/eye_calibration/pose_error", 10)
        self._pub_seg = self.create_publisher(Image, "/eye_calibration/segmentation", 0)
        self._sub_img = self.create_subscription(Image, "/image", self.callback_img, 0)
        self._sub_cam_info = self.create_subscription(
            CameraInfo, "/camera_info", self.callback_cam_info, 0
        )
        self._sub_rst = self.create_subscription(
            Empty, "/eye_calibration/restart", self.callback_rst, 0
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.add_on_set_parameters_callback(self.callback_params)

    def publish_target(self) -> None:
        """Publish the next marker wrt camera target pose."""
        tvec_mk_cam, quat_mk_cam = self._pose_target
        transform = Transform()
        transform.translation.x = float(tvec_mk_cam[0])
        transform.translation.y = float(tvec_mk_cam[1])
        transform.translation.z = float(tvec_mk_cam[2])
        transform.rotation.x = float(quat_mk_cam[0])
        transform.rotation.y = float(quat_mk_cam[1])
        transform.rotation.z = float(quat_mk_cam[2])
        transform.rotation.w = float(quat_mk_cam[3])

        msg = MultiDOFJointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names.append(str(self._ref_id))
        msg.points.append(
            MultiDOFJointTrajectoryPoint(transforms=[transform], velocities=[Twist()])
        )
        self._pub_target.publish(msg)

    def publish_perr(self) -> None:
        """Publish the estimated eye wrt camera pose error."""
        if len(self._poses_gt) != self._poses[0].shape[0]:
            return
        stamp = self.get_clock().now().to_msg()
        for i, (pose_gt, tvec, rmat) in enumerate(
            zip(self._poses_gt, self._poses[0], self._poses[1])
        ):
            rot_eye_cam = R.from_matrix(rmat.T.cpu().numpy())
            rot_cam_gt_eye = R.from_quat(pose_gt[3:])
            tvec_err = tvec.cpu().numpy() - pose_gt[:3]
            quat_err = (rot_cam_gt_eye * rot_eye_cam).as_quat()

            msg = PoseStamped()
            msg.header.stamp = stamp + i * 0.1
            msg.pose.position.x = float(tvec_err[0])
            msg.pose.position.y = float(tvec_err[1])
            msg.pose.position.z = float(tvec_err[2])
            msg.pose.orientation.x = float(quat_err[0])
            msg.pose.orientation.y = float(quat_err[1])
            msg.pose.orientation.z = float(quat_err[2])
            msg.pose.orientation.w = float(quat_err[3])
            self._pub_perr.publish(msg)

    def publish_seg(self, img: NDArray, mask: NDArray, header: Header) -> None:
        """Publish alpha-blended RGB + segmentation mask image.

        Args:
            rgb: Input RGB image. Shape is (H, W, 3) and dtype is `np.uint8`.
            mask: Output segmentation mask. Shape is (H, W) and dtype is `np.float32`.
            header: Header of the original Image message.
        """
        alpha = 0.6
        color = np.array((30, 144, 255))
        mask_img = mask[:, :, None] * color.reshape(1, 1, 3)  # (H, W, 3)
        seg_img = np.where(mask_img > 0, img * (1 - alpha) + mask_img * alpha, img).astype(np.uint8)
        msg = self._bridge.cv2_to_imgmsg(seg_img, encoding="rgb8", header=header)
        self._pub_seg.publish(msg)

    def callback_timer(self) -> None:
        if self._pose_target is None or self._pose_target_reached:
            return
        # extract marker pose
        transform = lookup_transform(self._frames["cam"], self._frames["marker"], self._tf_buffer)
        if transform is None:
            return
        pos, ori = transform.transform.translation, transform.transform.rotation
        tvec = np.array([pos.x, pos.y, pos.z])
        rot = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        # extract target pose
        tvec_mk_cam, quat_mk_cam = self._pose_target
        rot_target_inv = R.from_quat(quat_mk_cam)
        tvec_target = -rot_target_inv.apply(tvec_mk_cam, inverse=True)
        # compute pose error
        tvec_err = np.linalg.norm(tvec - tvec_target, axis=0)
        rot_err = (rot * rot_target_inv).magnitude()
        self._pose_target_reached = tvec_err < self._ref_ptol[0] and rot_err < self._ref_ptol[1]

    def callback_img(self, msg: Image) -> None:
        if len(self._silhouette) == self._n_view:
            return  # optimization done
        if self._pose_target is None:
            self._pose_target = self._get_target_pose()
            self.publish_target()
        if not self._pose_target_reached:
            return
        # store GT pose if available
        transform = lookup_transform(self._frames["cam"], self._frames["eye_gt"], self._tf_buffer)
        if transform is not None:
            pos, ori = transform.transform.translation, transform.transform.rotation
            self._poses_gt.append(np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]))
        # get segmentation mask
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self._sam.sample_points(img)
        mask = self._sam.segment(img)
        self.publish_seg(img, mask)
        # store silhouette and RGB
        self._silhouette.append(torch.from_numpy(mask).to(dtype=torch.float32, device=self._device))
        self._rgb.append(to_tensor(img).to(device=self._device))
        # perform optimization if all views are available
        if len(self._silhouette) == self._n_view:
            self._silhouette = torch.stack(self._silhouette, dim=0)
            self._rgb = torch.stack(self._rgb, dim=0)
            self._optimize()
            self.publish_perr()
        else:  # update target
            self._pose_target = self._get_target_pose()
            self._pose_target_reached = False

    def callback_cam_info(self, msg: CameraInfo) -> None:
        if self._cameras is None:
            camera = Camera.from_args(
                view_matrix=torch.eye(4),
                focal_x=msg.k[0],
                x0=msg.k[2] - msg.width / 2,
                y0=msg.k[5] - msg.header / 2,
                height=self._size,
                width=self._size,
                dtype=torch.float32,
            )
            self._cameras = Camera.cat([camera] * self._n_view).to(self._device)

    def callback_rst(self, msg: Empty) -> None:
        self._poses = self._init_poses()
        self._optim = self._init_optim(model=True)
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
                "dr.optim.loss.symmetry",
                "dr.optim.lr",
                "dr.optim.sched.eta_min",
            ] and (param.type_ != ParamType.DOUBLE or param.value < 0):
                failed(f"{param.name} must be double >= 0.")
                break
            if param.name == "dr.raster.backend" and param.type_ != ParamType.STRING:
                failed("dr.raster.backend must be a string.")
                break
            if param.name == "dr.optim.loss" and param.type_ != ParamType.STRING_ARRAY:
                failed("dr.optim.loss must be string array.")
                break
        if result.successful:
            self._optim = self._init_optim(renderer=True, optimizer=True)
        return result

    def _init_seg(self) -> SAM2:
        """Initialize the segmentation model."""
        variant = self.get_parameter("sam.variant").value
        prompt_n_pos = self.get_parameter("sam.prompt.n_pos").value
        prompt_n_neg = self.get_parameter("sam.prompt.n_neg").value
        return SAM2(
            var=variant,
            cfg=SAMPromptConfig(n_pos=prompt_n_pos, n_neg=prompt_n_neg),
            device=self._device,
        )

    def _init_poses(self) -> tuple[Tensor, Tensor]:
        """Initialize the target eye-to-camera poses (tvec, rmat)."""
        pose_params = {k: v.value for k, v in self.get_parameters_by_prefix("pose").items()}
        dist = pose_params["dist"]
        range_elev, range_azim = pose_params["range.elev"], pose_params["range.azim"]
        elev, azim = torch.meshgrid(
            torch.linspace(-range_elev, range_elev, pose_params["n_view_sqrt"]),
            torch.linspace(-range_azim, range_azim, pose_params["n_view_sqrt"]),
            indexing="xy",
        )
        elev, azim = elev.flatten(), azim.flatten()
        rmat, tvec = vc_kal.utils.look_at_view_transform(dist, elev, azim, device=self._device)
        return tvec, rmat

    def _init_optim(self, **kwargs) -> vc_kal.optim.EyePoseTextureOptimizer:
        """Initialize the DR-based eye pose and texture optimizer."""
        has_optim = hasattr(self, "_optim")
        mesh = (
            self._init_optim_mesh(self._n_view)
            if not has_optim or kwargs.get("mesh", False)
            else self._optim._mesh
        )
        model = (
            self._init_optim_model(self._n_view)
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
            symmetry_w, tan_norm_w = optim_params["loss.symmetry"], optim_params["loss.tan_norm"]
            lr, lr_sched_cfg = self._init_optim_lr_and_sched(optim_params)
        else:
            loss_fn = self._optim._loss_fn
            symmetry_w, tan_norm_w = self._optim._symmetry_w, self._optim._tan_norm_w
            lr, lr_sched_cfg = self._optim._lr, self._optim._sched_cfg
        return vc_kal.optim.EyePoseTextureOptimizer(
            mesh,
            model,
            mesh_renderer,
            loss_fn,
            symmetry_w=symmetry_w,
            tan_norm_w=tan_norm_w,
            lr=lr,
            lr_sched_cfg=lr_sched_cfg,
        )

    def _init_optim_mesh(self, n_view: int) -> vc_kal.mesh.ObjMesh:
        """Intialize the obj mesh used by the optimizer."""
        # load mesh from obj
        path = self.get_parameter("dr.mesh.path").value
        mesh = vc_kal.mesh.ObjMesh(path, with_materials=True, with_normals=True, n_rep=n_view)
        # ensure that vertex_features is not None
        n_face = mesh.mesh.vertices.shape[1]
        mesh.mesh.vertex_features = torch.zeros([n_view, n_face, 0])
        # update UVs and texture to Kaolin conventions
        mesh.mesh.face_uvs = 1 - mesh.mesh.face_uvs
        mesh.mesh.materials[0][0]["map_Kd"] = (
            mesh.mesh.materials[0][0]["map_Kd"].float().permute(2, 0, 1) / 255.0
        )
        return mesh.to(device=self._device)

    def _init_optim_model(self, n_view: int) -> common.model.EyePoseTextureModel:
        """Initialize the eye pose and texture model used by the optimizer."""
        model_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.model").items()}
        model_type = model_params["type"]
        assert model_type in ["simple", "mipmap", "hashenc"]
        kwargs = {
            "pos": self._poses[0],
            "z_dir": self._poses[1][..., -1],
            "res": model_params["res"],
            "n_view": n_view,
            "scale": model_params["scale"],
        }
        match model_type:
            case "simple":
                model = common.model.EyePoseTextureModel(text_init=torch.zeros(3), **kwargs)
            case "mipmap":
                model = common.model.EyePoseTextureMipmapModel(
                    text_init=torch.full((3,), 0.5), n_level=model_params["n_level"], **kwargs
                )
            case "hashenc":
                kwargs.pop("res")
                enc_cfg = common.model.HashEncoder2DCfg(
                    finest_res=model_params["res"],
                    n_level=model_params["n_level"],
                    log2_hashmap_size=model_params["log2_hsz"],
                )
                model = common.model.EyePoseTextureHashEncoderModel(
                    enc_cfg=enc_cfg, mlp_n_layer=model_params["mlp_n_layer"], **kwargs
                )
        return model.to(device=self._device)

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
                kwargs=[{}, {"inner_fn_name": optim_params["loss"][1]}],
            )
        )

    def _init_optim_lr_and_sched(self, optim_params: dict[str, Any]):
        """Initialize the learning rate and LR scheduler config used by the optimzier."""
        lr = optim_params["lr"]
        lr_sched_cfg = vc_kal.optim.EyePoseTextureOptimizer.LRSchedulerCfg(
            cls=CosineAnnealingWarmRestarts,
            kwargs={
                "T_0": optim_params["sched.T_0"],
                "T_mult": optim_params["sched.T_mult"],
                "eta_min": optim_params["sched.eta_min"],
            },
        )
        return lr, lr_sched_cfg

    def _get_target_pose(self) -> tuple[NDArray, NDArray]:
        """Compute the next marker wrt camera target pose."""
        n_curr = len(self._rgb)
        tvec, rmat = self._poses[0][n_curr], self._poses[1][n_curr]
        tvec_eye_cam = (-rmat.T @ tvec).cpu().numpy()
        rot_eye_cam = R.from_matrix(rmat.T.cpu().numpy())
        tvec_mk_eye = self._ref_pose[:3]
        rot_mk_eye = R.from_quat(self._ref_pose[3:], scalar_first=True)
        tvec_mk_cam = rot_mk_eye.apply(tvec_eye_cam) + tvec_mk_eye
        quat_mk_cam = (rot_mk_eye * rot_eye_cam).as_quat()
        return tvec_mk_cam, quat_mk_cam

    def _optimize(self) -> None:
        """Run optimization for eye pose and texture parameters."""
        n_iter = self.get_parameter("dr.optim.n_iter.full").value
        n_iter_text = self.get_parameter("dr.optim.n_iter.text").value
        # prepare target and cameras
        center = torch.stack([self._get_centroid(sil) for sil in self._silhouette], dim=0)
        center = center.mean(dim=0).long()
        target = self._get_target(center)
        self._cameras.x0 -= center[1] - self._size / 2
        self._cameras.y0 -= center[0] - self._size / 2
        # run silhouette-only optimization
        self._optimize_silhouette(target)
        # run main optimization
        tvec, rmat, texture = self._optim.optimize(
            target, n_iter, n_iter_text, cameras=self._cameras
        )
        self._poses = (tvec, rmat)
        # render final output with texture and poses
        final = self._optim._renderer(
            self._optim._mesh({}, texture=texture), cameras=self._cameras, R=rmat, T=tvec
        ).detach()
        self._store_outputs(texture, final)

    def _optimize_silhouette(self, target: Tensor) -> None:
        """Run silhouette-only optimizer to reduce pose errors before main optimization."""
        n_iter = self.get_parameter("dr.optim.n_iter.init").value
        # replace loss function, LR and LR scheduler
        self._optim._loss_fn = torch.compile(
            common.losses.build_combined_loss_fn(
                ["mse_loss"], [slice(1)], device=self._device, reduction="mean"
            )
        )
        self._optim._lr = 1e-2
        self._optim._sched_cfg = vc_kal.optim.EyePoseTextureOptimizer.LRSchedulerCfg(
            cls=CosineAnnealingLR, kwargs={"T_max": n_iter, "eta_min": 1e-6}
        )
        # run optimizer to update model's pose parameters
        self._optim.optimize(target, n_iter=n_iter, n_iter_text=0, cameras=self._cameras)
        # reset optimizer back to its initial state
        self._optim = self._init_optim(optimizer=True)

    def _get_target(self, center: LongTensor) -> Tensor:
        """Get the target image from stored silhouette and RGB images."""
        silhouette = self._crop_img(self._silhouette.unsqueeze(-1), center)
        rgb = self._crop_img(self._rgb * self._silhouette.unsqueeze(-1), center)
        return torch.cat([silhouette, rgb], dim=-1)

    def _crop_img(self, img: Tensor, center: LongTensor) -> Tensor:
        """Crop input image at the given center and size to prepare for rendering."""
        H, W = img.shape[-3:-1]
        top = max(int(center[0] - self._size / 2), 0)
        left = max(int(center[1] - self._size / 2), 0)
        right = min(left + self._size, W)
        bottom = min(top + self._size, H)
        return img[..., top:bottom, left:right, :]

    def _store_outputs(self, texture: Tensor, final: Tensor) -> None:
        """Store the output texture and final renders."""
        path = self.get_parameter("output_path").value
        texture = texture.permute(2, 0, 1)
        final = final.permute(0, 3, 1, 2)
        save_image(texture, f"{path}/texture.png")
        save_image(final, f"{path}/final.png", nrow=self._n_view)

    def _reset(self) -> None:
        """Reset all attributes that are initialized from callbacks."""
        self._cameras = None
        self._silhouette = []
        self._rgb = []
        self._poses_gt = []
        self._pose_target = None
        self._pose_target_reached = False

    @staticmethod
    def _get_centroid(mask: Tensor) -> LongTensor | None:
        """Compute the centroid (row, col) of a boolean segmentation (H, W) mask."""
        indices = torch.nonzero(mask).float()
        if indices.shape[0] == 0:
            return None  # empty mask
        return indices.mean(dim=0).long()


def main(args=None):
    rclpy.init(args=args)
    eye_calibration = EyeCalibration()
    rclpy.spin(eye_calibration)
    eye_calibration.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
