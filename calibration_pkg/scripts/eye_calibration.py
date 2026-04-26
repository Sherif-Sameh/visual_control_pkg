#!/usr/bin/env python3

"""
ROS node for performing eye texture calibration using differentiable rendering (Kaolin)
"""

import logging
import pickle
from copy import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from kaolin.render.camera import Camera
from numpy.typing import NDArray
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty, Header
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision.utils import save_image

import vc_core.dr.common as common
import vc_core.dr.kaolin as vc_kal
import vc_core.utils.geometry.pose as pose_utils
from vc_core.segmentation.sam import SAM2, SAMPromptConfig
from vc_core.utils.ros.tf2 import lookup_transform

torch.set_float32_matmul_precision("high")
logging.getLogger("kaolin.rep.surface_mesh").setLevel(logging.ERROR)


class EyeCalibration(Node):
    """ROS node for performing eye texture calibration using differentiable rendering (Kaolin)."""

    EYE_ROT_OFFSET = R.from_quat([1.0, 0.0, 0.0, 0.0])

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
        self.declare_parameter("pose.range.elev", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("pose.range.azim", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("dr.mesh.path", rclpy.Parameter.Type.STRING)
        self.declare_parameter("dr.mesh.scale", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.mesh.elev_lim", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("dr.mesh.azim_lim", rclpy.Parameter.Type.DOUBLE_ARRAY)
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
        self._sam_prompts = self._init_seg_prompts()
        self._pose_home = None
        self._poses = self._init_poses()
        self._optim = self._init_optim()
        self._reset()

        # Initialize ROS attributes
        self._timer = self.create_timer(0.5, self.callback_timer)
        self._pub_target = self.create_publisher(
            PoseStamped,
            "/eye_calibration/command",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._pub_perr = self.create_publisher(PoseStamped, "/eye_calibration/pose_error", 10)
        self._pub_seg = self.create_publisher(Image, "/eye_calibration/segmentation", 1)
        self._sub_img = self.create_subscription(Image, "/image", self.callback_img, 1)
        self._sub_cam_info = self.create_subscription(
            CameraInfo, "/camera_info", self.callback_cam_info, 1
        )
        self._sub_rst = self.create_subscription(
            Empty, "/eye_calibration/restart", self.callback_rst, 1
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.add_on_set_parameters_callback(self.callback_params)

    def publish_target(self, header: Header) -> None:
        """Publish the next marker wrt camera target pose."""
        tvec_mk_cam, quat_mk_cam = self._pose_target
        msg = PoseStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self._frames["marker"]
        msg.pose.position.x = float(tvec_mk_cam[0])
        msg.pose.position.y = float(tvec_mk_cam[1])
        msg.pose.position.z = float(tvec_mk_cam[2])
        msg.pose.orientation.x = float(quat_mk_cam[0])
        msg.pose.orientation.y = float(quat_mk_cam[1])
        msg.pose.orientation.z = float(quat_mk_cam[2])
        msg.pose.orientation.w = float(quat_mk_cam[3])
        self._pub_target.publish(msg)

    def publish_tf(self, header: Header) -> None:
        """Publish the next marker wrt camera target TF transform."""
        tvec_mk_cam, quat_mk_cam = self._pose_target
        transform = TransformStamped()
        transform.header.stamp = header.stamp
        transform.header.frame_id = self._frames["marker"]
        transform.child_frame_id = f"{self._frames['cam']}:{self._ref_id}"
        transform.transform.translation.x = float(tvec_mk_cam[0])
        transform.transform.translation.y = float(tvec_mk_cam[1])
        transform.transform.translation.z = float(tvec_mk_cam[2])
        transform.transform.rotation.x = float(quat_mk_cam[0])
        transform.transform.rotation.y = float(quat_mk_cam[1])
        transform.transform.rotation.z = float(quat_mk_cam[2])
        transform.transform.rotation.w = float(quat_mk_cam[3])
        self._tf_broadcaster.sendTransform(transform)

    def publish_perr(self) -> None:
        """Publish the estimated eye wrt camera pose error."""
        if len(self._poses_gt) != self._poses[0].shape[0]:
            return
        # compute per-pose errors
        tvec_err, rot_err = [], []
        for pose_gt, tvec, rmat in zip(self._poses_gt, self._poses[0], self._poses[1]):
            tvec_cam_eye, rot_cam_eye = self._apply_rot_offset(tvec, rmat)
            rot_cam_gt_eye = R.from_quat(pose_gt[3:])
            tvec_err.append(tvec_cam_eye - pose_gt[:3])
            rot_err.append(rot_cam_gt_eye * rot_cam_eye.inv())
        # average pose errors
        tvec_err = np.abs(np.stack(tvec_err, axis=0)).mean(axis=0)
        quat_err = R.from_euler("x", R.concatenate(rot_err).magnitude().mean()).as_quat()

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

    def publish_seg(self, img: NDArray, mask: NDArray) -> None:
        """Publish alpha-blended RGB + segmentation mask image.

        Args:
            rgb: Input RGB image. Shape is (H, W, 3) and dtype is `np.uint8`.
            mask: Output segmentation mask. Shape is (H, W) and dtype is `np.float32`.
        """
        alpha = 0.6
        color = np.array((30, 144, 255))
        mask_img = mask[:, :, None] * color.reshape(1, 1, 3)  # (H, W, 3)
        seg_img = np.where(mask_img > 0, img * (1 - alpha) + mask_img * alpha, img).astype(np.uint8)
        msg = self._bridge.cv2_to_imgmsg(
            seg_img, encoding="rgb8", header=Header(stamp=self.get_clock().now().to_msg())
        )
        self._pub_seg.publish(msg)

    def callback_timer(self) -> None:
        tracking = self._pose_target is not None and not self._pose_target_reached
        if self._pose_home is not None and not tracking:
            return
        # extract marker pose
        transform = lookup_transform(self._frames["cam"], self._frames["marker"], self._tf_buffer)
        if transform is None:
            return
        tvec, rot = pose_utils.from_transform_gm(transform.transform)
        if self._pose_home is None:
            tvec_home, rot_home = pose_utils.pose_inv((tvec, rot))
            self._pose_home = (tvec_home, rot_home.as_quat())
        if tracking:
            # extract target pose
            tvec_mk_cam, quat_mk_cam = self._pose_target
            rot_target_inv = R.from_quat(quat_mk_cam)
            tvec_target = -rot_target_inv.apply(tvec_mk_cam, inverse=True)
            # compute pose error
            tvec_err = np.linalg.norm(tvec - tvec_target, axis=0)
            rot_err = (rot * rot_target_inv).magnitude()
            self._pose_target_reached = tvec_err < self._ref_ptol[0] and rot_err < self._ref_ptol[1]

    def callback_img(self, msg: Image) -> None:
        n_curr = len(self._silhouette)
        if n_curr == self._n_view:
            return  # optimization done
        if self._pose_target is None:
            self._pose_target = self._get_target_pose()
            self.publish_target(msg.header)
        self.publish_tf(msg.header)
        if not self._pose_target_reached:
            return
        # store GT pose if available
        transform = lookup_transform(self._frames["cam"], self._frames["eye_gt"], self._tf_buffer)
        if transform is not None:
            pos, ori = transform.transform.translation, transform.transform.rotation
            self._poses_gt.append(np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]))
        # get segmentation mask
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if len(self._sam_prompts) < self._n_view:
            self._sam.sample_points(img)
            self._sam_prompts.append((copy(self._sam.point_prompts), copy(self._sam.label_prompts)))
            if len(self._sam_prompts) == self._n_view:
                self._store_prompts()
        else:
            self._sam.point_prompts = self._sam_prompts[n_curr][0]
            self._sam.label_prompts = self._sam_prompts[n_curr][1]
        mask = self._sam.segment(img)
        self.publish_seg(img, mask)
        # store silhouette and RGB
        self._silhouette.append(
            torch.from_numpy(mask).contiguous().to(dtype=torch.float32, device=self._device)
        )
        self._rgb.append(
            torch.from_numpy(img).contiguous().to(dtype=torch.float32, device=self._device).div(255)
        )
        # perform optimization if all views are available
        if (n_curr + 1) == self._n_view:
            self._silhouette = torch.stack(self._silhouette, dim=0)
            self._rgb = torch.stack(self._rgb, dim=0)
            self._optimize()
            self.publish_perr()
            self._pose_target = self._pose_home
        else:  # update target
            self._pose_target = self._get_target_pose()
            self._pose_target_reached = False
        self.publish_target(msg.header)

    def callback_cam_info(self, msg: CameraInfo) -> None:
        if self._cameras is None:
            camera = Camera.from_args(
                view_matrix=torch.eye(4),
                focal_x=msg.k[0],
                x0=msg.k[2] - msg.width / 2,
                y0=msg.k[5] - msg.height / 2,
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
                "dr.optim.loss.tan_norm",
                "dr.optim.lr",
                "dr.optim.sched.eta_min",
            ] and (param.type_ != ParamType.DOUBLE or param.value < 0):
                failed(f"{param.name} must be double >= 0.")
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

    def _init_seg_prompts(self) -> list[tuple[list[tuple[int, int]], list[int]]]:
        """Initialize segmentation prompts from pickle file if it exists."""
        path = Path(self.get_parameter("output_path").value).parent / "prompts.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return []

    def _init_poses(self) -> tuple[Tensor, Tensor]:
        """Initialize the target eye-to-camera poses (tvec, rmat)."""
        pose_params = {k: v.value for k, v in self.get_parameters_by_prefix("pose").items()}
        dist = pose_params["dist"]
        range_elev, range_azim = pose_params["range.elev"], pose_params["range.azim"]
        elev, azim = torch.meshgrid(
            torch.linspace(range_elev[0], range_elev[1], pose_params["n_view_sqrt"]),
            torch.linspace(range_azim[0], range_azim[1], pose_params["n_view_sqrt"]),
            indexing="xy",
        )
        elev_sorted, azim_sorted = [], []
        for i, (row_e, row_a) in enumerate(zip(elev, azim)):
            if i % 2 == 1:
                row_e = torch.flip(row_e, dims=[0])
                row_a = torch.flip(row_a, dims=[0])
            elev_sorted.append(row_e)
            azim_sorted.append(row_a)
        elev, azim = torch.cat(elev_sorted, 0), torch.cat(azim_sorted, 0)
        rmat, tvec = vc_kal.utils.look_at_view_transform(dist, elev, azim, device=self._device)
        return tvec, rmat

    def _init_optim(self, **kwargs) -> vc_kal.optim.EyePoseTextureOptimizer:
        """Initialize the DR-based eye pose and texture optimizer."""
        has_optim = hasattr(self, "_optim")
        mesh = (
            self._init_optim_mesh()
            if not has_optim or kwargs.get("mesh", False)
            else self._optim._mesh
        )
        model = (
            self._init_optim_model(mesh)
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

    def _init_optim_mesh(self) -> vc_kal.mesh.EyeObjMesh:
        """Intialize the obj mesh used by the optimizer."""
        # load mesh from obj
        mesh_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.mesh").items()}
        mesh = vc_kal.mesh.EyeObjMesh(
            mesh_params["path"],
            n_rep=self._n_view,
            elev_lim=mesh_params["elev_lim"],
            azim_lim=mesh_params["azim_lim"],
        )
        mesh.mesh.vertices *= mesh_params["scale"]
        self._mesh_offset = torch.tensor(
            [0.0, 0.0, mesh.mesh.vertices[..., -1].amax()], device=self._device
        )
        # ensure that vertex_features is not None
        n_face = mesh.mesh.vertices.shape[1]
        mesh.mesh.vertex_features = torch.zeros([self._n_view, n_face, 0])
        # update UVs and texture to Kaolin conventions
        mesh.mesh.face_uvs = 1 - mesh.mesh.face_uvs
        mesh.mesh.materials[0][0]["map_Kd"] = (
            mesh.mesh.materials[0][0]["map_Kd"].permute(2, 0, 1).contiguous().float().div(255)
        )
        return mesh.to(device=self._device)

    def _init_optim_model(self, mesh: vc_kal.mesh.EyeObjMesh) -> common.model.EyePoseTextureModel:
        """Initialize the eye pose and texture model used by the optimizer."""
        model_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.model").items()}
        model_type = model_params["type"]
        assert model_type in ["simple", "mipmap", "hashenc"]
        kwargs = {
            "pos": self._poses[0],
            "z_dir": self._poses[1][..., -1],
            "res": model_params["res"],
            "text_ref": mesh.mesh.materials[0][0]["map_Kd"],
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
        tvec_eye_cam, rot_eye_cam = self._apply_rot_offset(-rmat.T @ tvec, rmat.T)
        tvec_mk_eye = self._ref_pose[:3]
        rot_mk_eye = R.from_quat(self._ref_pose[3:], scalar_first=True)
        tvec_mk_cam = rot_mk_eye.apply(tvec_eye_cam) + tvec_mk_eye
        quat_mk_cam = (rot_mk_eye * rot_eye_cam).as_quat()
        return tvec_mk_cam, quat_mk_cam

    def _apply_rot_offset(self, tvec: Tensor, rmat: Tensor) -> tuple[NDArray, R]:
        """Apply the fixed rotation offset between for eye wrt camera poses.

        This offset is caused by two rotations. Firstly between the OpenGL and ROS camera
        conventions. Secondly between the eye mesh's coordinate frame and the eye frame we use.
        Both are corrected by a rotation of 180deg around the X-axis.
        """
        tvec_new = self.EYE_ROT_OFFSET.apply(tvec.cpu().numpy())
        rot_new = self.EYE_ROT_OFFSET * R.from_matrix(rmat.cpu().numpy()) * self.EYE_ROT_OFFSET
        return tvec_new, rot_new

    def _optimize(self) -> None:
        """Run optimization for eye pose and texture parameters."""
        n_iter = self.get_parameter("dr.optim.n_iter.full").value
        n_iter_text = self.get_parameter("dr.optim.n_iter.text").value
        # get target and initial renders
        target = self._get_target()
        initial = self._optim._renderer(
            self._optim._mesh.mesh, cameras=self._cameras, R=self._poses[1], T=self._poses[0]
        ).detach()
        # run silhouette-only optimization and get intermediate renders
        tvec, rmat = self._optimize_silhouette(target)
        interm = self._optim._renderer(
            self._optim._mesh.mesh, cameras=self._cameras, R=rmat, T=tvec
        ).detach()
        # run main optimization
        tvec, rmat, texture = self._optim.optimize(
            target, n_iter=n_iter, n_iter_text=n_iter_text, cameras=self._cameras
        )
        tvec = tvec + rmat @ self._mesh_offset
        self._poses = (tvec, rmat)
        # render final output with texture and poses
        final = self._optim._renderer(
            self._optim._mesh({}, texture=texture), cameras=self._cameras, R=rmat, T=tvec
        ).detach()
        self._store_outputs(texture, initial, interm, final, target)
        torch.cuda.empty_cache()

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
        tvec, rmat, _ = self._optim.optimize(
            target, n_iter=n_iter, n_iter_text=0, cameras=self._cameras
        )
        # reset optimizer back to its initial state
        self._optim = self._init_optim(optimizer=True)
        return tvec, rmat

    def _get_target(self) -> Tensor:
        """Get the target image from stored silhouette and RGB images."""
        silhouette = self._crop_img(self._silhouette.unsqueeze(-1))
        rgb = self._crop_img(self._rgb * self._silhouette.unsqueeze(-1))
        return torch.cat([silhouette, rgb], dim=-1)

    def _crop_img(self, img: Tensor) -> Tensor:
        """Crop input image at its center to prepare for rendering."""
        H, W = img.shape[-3:-1]
        top = max(int((H - self._size) / 2), 0)
        left = max(int((W - self._size) / 2), 0)
        right = min(left + self._size, W)
        bottom = min(top + self._size, H)
        return img[..., top:bottom, left:right, :]

    def _store_prompts(self) -> None:
        """Store the collected segmentation prompts."""
        path = Path(self.get_parameter("output_path").value).parent / "prompts.pkl"
        with open(path, "wb") as f:
            pickle.dump(self._sam_prompts, f)

    def _store_outputs(
        self, texture: Tensor, initial: Tensor, interm: Tensor, final: Tensor, target: Tensor
    ) -> None:
        """Store the output texture, renders and targets."""
        path = Path(self.get_parameter("output_path").value)
        n_view_sqrt = self.get_parameter("pose.n_view_sqrt").value
        path.mkdir(parents=True, exist_ok=True)
        save_image(texture, f"{path}/texture.png")
        args = (initial, interm, final, target)
        labels = ("initial", "interm", "final", "target")
        for arg, label in zip(args, labels):
            arg = arg.permute(0, 3, 1, 2)
            save_image(arg[:, 1:], f"{path}/{label}.png", nrow=n_view_sqrt, pad_value=1.0)
        self.get_logger().info(f"Outputs stored at {path}")

    def _reset(self) -> None:
        """Reset all attributes that are initialized from callbacks."""
        self._cameras = None
        self._silhouette = []
        self._rgb = []
        self._poses_gt = []
        self._pose_target = None
        self._pose_target_reached = False


def main(args=None):
    rclpy.init(args=args)
    eye_calibration = EyeCalibration()
    rclpy.spin(eye_calibration)
    eye_calibration.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
