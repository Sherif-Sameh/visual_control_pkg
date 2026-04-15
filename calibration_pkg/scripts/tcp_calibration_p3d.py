#!/usr/bin/env python3

"""
ROS node for performing TCP to camera calibration using differentiable rendering (PyTorch3D)
"""

from typing import Any, Callable

import numpy as np
import pytorch3d.renderer as renderer
import rclpy
import torch
import torchvision.transforms.functional as VF
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from numpy.typing import NDArray
from pytorch3d.renderer.mesh.shader import SoftDepthShader
from pytorch3d.utils import cameras_from_opencv_projection
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty
from std_srvs.srv import SetBool
from tf2_ros import TransformBroadcaster
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR

import vc_core.dr.pytorch3d as vc_pytorch3d
from vc_core.dr.common.losses import build_combined_loss_fn
from vc_core.dr.common.model import CylinderSplitParamModel
from vc_core.dr.pytorch3d import CylinderMultiLROptimizer
from vc_core.loggers import MemoryLogger
from vc_core.segmentation.sam import SAM2, SAMPromptConfig


class TcpCalibrationP3d(Node):
    """ROS node for performing TCP to camera calibration using differentiable rendering (PyTorch3D)."""

    def __init__(self):
        super().__init__("tcp_calibration_p3d")

        # Declare ROS parameters
        self.declare_parameter("pose_gt", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("img_center", rclpy.Parameter.Type.INTEGER_ARRAY)
        self.declare_parameter("sam.variant", rclpy.Parameter.Type.STRING)
        self.declare_parameter("sam.prompt.type", rclpy.Parameter.Type.STRING)
        self.declare_parameter("sam.prompt.n_pos", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("sam.prompt.n_neg", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.mesh.radius", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.mesh.height", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.mesh.resolution", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.mesh.split", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.init.dist", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.init.elev", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.init.azim", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.model.radius", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("dr.model.height", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("dr.model.scale", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.shader.modalities", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("dr.shader.sigma", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.shader.gamma", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.shader.zfar", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.raster.size", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.raster.n_faces", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.optim.n_rep", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("dr.optim.sigma", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("dr.optim.loss", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("dr.optim.loss.weights", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("dr.optim.lr.min", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.optim.lr.max", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.optim.sched", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("dr.optim.sched.eta_min", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("dr.optim.n_iter", rclpy.Parameter.Type.INTEGER)

        # Initialize non-ROS class attributes
        pose_gt = self.get_parameter("pose_gt").value
        has_depth = "depth" in self.get_parameter("dr.shader.modalities").value
        n_iter = self.get_parameter("dr.optim.n_iter").value
        if not torch.cuda.is_available():
            self.get_logger().warn("CUDA is not available. DR is very slow running on CPU.")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pose_gt = np.array(pose_gt) if len(pose_gt) == 7 else None
        self._img_center = self.get_parameter("img_center").value
        self._prompt_type = self.get_parameter("sam.prompt.type").value
        self._size = self.get_parameter("dr.raster.size").value
        self._bridge = CvBridge()
        self._memory_logger = MemoryLogger(n_log=n_iter, filter="loss")
        self._sam = self._init_seg()
        self._optim = self._init_optim()
        self._optimize = self._build_optimize_fn()
        self._reset()

        # Initialize ROS attributes
        self._timer = self.create_timer(0.25, self.callback_timer)
        self._pub_perr = self.create_publisher(PoseStamped, "/tcp_calibration_p3d/pose_error", 10)
        self._pub_seg = self.create_publisher(Image, "/tcp_calibration_p3d/segmentation", 0)
        self._sub_img = self.create_subscription(Image, "/image", self.callback_img, 0)
        if has_depth:
            self._sub_depth = self.create_subscription(Image, "/depth", self.callback_depth, 0)
        self._sub_cam_info = self.create_subscription(
            CameraInfo, "/camera_info", self.callback_cam_info, 0
        )
        self._sub_rst = self.create_subscription(
            Empty, "/tcp_calibration_p3d/restart", self.callback_rst, 0
        )
        self._srv_trgr = self.create_service(
            SetBool, "/tcp_calibration_p3d/trigger", self.callback_trgr
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self.add_on_set_parameters_callback(self.callback_params)

    def publish_perr(self, tvec: Tensor, rmat: Tensor, loss: float) -> None:
        """Publish the estimated object-to-camera pose error.

        Args:
            tvec: Estimated translation vector. Shape is (3,).
            rmat: Estimation rotation matrix. Shape is (3, 3).
            loss: Loss value at the final iteration. Used if ground-truth pose is not available.
        """
        if self._pose_gt is not None:
            rot_cam_obj = R.from_matrix(rmat.cpu().numpy())
            rot_cam_obj_gt = R.from_quat(self._pose_gt[3:], scalar_first=True)
            tvec_err = tvec.cpu().numpy() - self._pose_gt[:3]
            quat_err = (rot_cam_obj_gt.inv() * rot_cam_obj).as_quat()
        else:
            tvec_err = np.array([loss] * 3)
            quat_err = R.identity().as_quat()

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

    def publish_tf(self, tvec: Tensor, rmat: Tensor) -> None:
        """Publish the estimated object-to-camera transform to the TF tree.

        Args:
            tvec: Estimated translation vector. Shape is (3,).
            rmat: Estimation rotation matrix. Shape is (3, 3).
        """
        tvec_np = tvec.cpu().numpy()
        quat_np = R.from_matrix(rmat.cpu().numpy()).as_quat()

        transform = TransformStamped()
        transform.header = self._header
        transform.child_frame_id = "tcp_e"
        transform.transform.translation.x = float(tvec_np[0])
        transform.transform.translation.y = float(tvec_np[1])
        transform.transform.translation.z = float(tvec_np[2])
        transform.transform.rotation.x = float(quat_np[0])
        transform.transform.rotation.y = float(quat_np[1])
        transform.transform.rotation.z = float(quat_np[2])
        transform.transform.rotation.w = float(quat_np[3])
        self._tf_broadcaster.sendTransform(transform)

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
        msg = self._bridge.cv2_to_imgmsg(seg_img, encoding="rgb8", header=self._header)
        self._pub_seg.publish(msg)

    def callback_timer(self) -> None:
        if self._pose is None:
            tvec, rmat, loss = self._optimize()
            if self._pose is not None:
                self.publish_perr(tvec, rmat, loss)
        else:
            tvec, rmat = self._pose
        if self._header is not None:
            self.publish_tf(tvec, rmat)

    def callback_img(self, msg: Image) -> None:
        self._header = msg.header
        if self._silhoutte is not None:
            return  # don't override existing
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self._sam.sample_points(img) if self._prompt_type == "points" else self._sam.sample_box(img)
        mask = self._sam.segment(img)
        self._silhoutte = torch.from_numpy(mask).to(dtype=torch.float32, device=self._device)
        self.publish_seg(img, mask)

    def callback_depth(self, msg: Image) -> None:
        if self._depth is not None:
            return  # don't override existing
        depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if msg.encoding == "16UC1":
            depth = depth.astype(np.float32) * 1e-3  # convert mm to meters
        self._depth = torch.from_numpy(depth).to(dtype=torch.float32, device=self._device)

    def callback_cam_info(self, msg: CameraInfo) -> None:
        if self._camera is None:
            K = np.array(msg.k).reshape(1, 3, 3)
            K[0, 0, 2] -= self._img_center[1] - self._size / 2
            K[0, 1, 2] -= self._img_center[0] - self._size / 2
            self._camera = cameras_from_opencv_projection(
                R=torch.eye(3).view(1, 3, 3),
                tvec=torch.zeros((1, 3)),
                camera_matrix=torch.from_numpy(K).to(dtype=torch.float32),
                image_size=torch.tensor([[self._size] * 2]),
            ).to(device=self._device)

    def callback_rst(self, msg: Empty) -> None:
        self._camera = None
        self._depth = None
        self._pose = None
        self._optim = self._init_optim(model=True)

    def callback_trgr(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        if req.data:  # reset everything including target inputs
            self._reset()
        else:  # reset computed pose only
            self._pose = None
        self._optim = self._init_optim(model=True)
        res.success = True
        return res

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
            if param.name in ["dr.shader.sil.sigma", "dr.optim.lr", "dr.optim.sched.eta_min"] and (
                param.type_ != ParamType.DOUBLE or param.value < 0
            ):
                failed(f"{param.name} must be double >= 0.")
                break
            if (
                param.name == "dr.optim.sigma"
                and param.type_ != ParamType.DOUBLE_ARRAY
                and len(param.value) != 2
            ):
                failed("dr.optim.sigma must be a double array of length 2.")
                break
            if param.name == "dr.optim.loss" and param.type_ != ParamType.STRING_ARRAY:
                failed("dr.optim.loss must be string array.")
                break
        if result.successful:
            self._pose = None
            self._optim = self._init_optim(model=True, renderer=True, optimizer=True)
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

    def _init_optim(self, **kwargs) -> CylinderMultiLROptimizer:
        """Initialize the DR-based optimizer."""
        has_optim = hasattr(self, "_optim")
        n_rep = self.get_parameter("dr.optim.n_rep").value
        mesh = (
            self._init_optim_mesh(n_rep)
            if not has_optim or kwargs.get("mesh", False)
            else self._optim._mesh
        )
        model = (
            self._init_optim_model(n_rep)
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
            lr, lr_sched_cfg = self._init_optim_lr_and_sched(optim_params, n_rep)
        else:
            loss_fn = self._optim._loss_fn
            lr, lr_sched_cfg = self._optim._lr, self._optim._sched_cfg
        return CylinderMultiLROptimizer(
            mesh, model, mesh_renderer, loss_fn, lr=lr, lr_sched_cfg=lr_sched_cfg
        )

    def _init_optim_mesh(self, n_rep: int) -> vc_pytorch3d.CylinderMesh:
        """Intialize the cylinder mesh used by the optimizer."""
        mesh_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.mesh").items()}
        return vc_pytorch3d.CylinderMesh(n_rep=n_rep, **mesh_params).to(device=self._device)

    def _init_optim_model(self, n_rep: int) -> CylinderSplitParamModel:
        """Initialize the cylinder model used by the optimizer."""
        radius = self.get_parameter("dr.mesh.radius").value
        height = self.get_parameter("dr.mesh.height").value
        sigma = self.get_parameter("dr.optim.sigma").value
        init_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.init").items()}
        model_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.model").items()}
        rmat, tvec = renderer.look_at_view_transform(**init_params)
        return CylinderSplitParamModel(
            pos=tvec + torch.normal(0, sigma[0], size=(n_rep, 3)),
            z_dir=rmat[:, :, 2] + torch.normal(0, sigma[1], size=(n_rep, 3)),
            n_rep=n_rep,
            radius=torch.tensor([radius]) if model_params["radius"] else None,
            height=torch.tensor([height]) if model_params["height"] else None,
            scale=model_params["scale"],
        ).to(device=self._device)

    def _init_optim_renderer(self) -> renderer.MeshRenderer:
        """Initialize the renderer used by the optimizer."""
        shader_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.shader").items()}
        raster_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.raster").items()}
        blend_params = renderer.BlendParams(
            sigma=shader_params["sigma"], gamma=shader_params["gamma"]
        )
        raster_settings = renderer.RasterizationSettings(
            image_size=raster_params["size"],
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=raster_params["n_faces"],
            perspective_correct=True,
        )
        shaders = []
        if "silhouette" in shader_params["modalities"]:
            shaders.append(vc_pytorch3d.SoftSilhouetteShader(blend_params=blend_params))
        if "depth" in shader_params["modalities"]:
            shaders.append(SoftDepthShader(blend_params=blend_params))
        return renderer.MeshRenderer(
            rasterizer=renderer.MeshRasterizer(raster_settings=raster_settings),
            shader=vc_pytorch3d.ComposeShader(shaders),
        ).to(device=self._device)

    def _init_optim_loss_fn(
        self, optim_params: dict[str, Any]
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Initialize the loss function used by the optimizer."""
        modalities = self.get_parameter("dr.shader.modalities").value
        losses, ranges, weights = [], [], []
        if "silhouette" in modalities:
            losses.append(optim_params["loss"][0])
            ranges.append(slice(1))
            weights.append(optim_params["loss.weights"][0])
        if "depth" in modalities:
            losses.append(optim_params["loss"][1])
            ranges.append(slice(1) if len(losses) == 1 else slice(1, 2))
            weights.append(optim_params["loss.weights"][1])
        return torch.compile(
            build_combined_loss_fn(
                losses,
                ranges,
                weights=weights,
                device=self._device,
                reduction="mean",
                dim=(1, 2, 3),
            )
        )

    def _init_optim_lr_and_sched(self, optim_params: dict[str, Any], n_rep: int):
        """Initialize the learning rate and LR scheduler config used by the optimzier."""
        lr_min, lr_max = optim_params["lr.min"], optim_params["lr.max"]
        lr = np.exp(np.random.uniform(np.log(lr_min), np.log(lr_max), n_rep)).tolist()
        lr_sched_cfg = (
            CylinderMultiLROptimizer.LRSchedulerCfg(
                cls=CosineAnnealingLR,
                kwargs={"T_max": optim_params["n_iter"], "eta_min": optim_params["sched.eta_min"]},
            )
            if optim_params["sched"]
            else None
        )
        return lr, lr_sched_cfg

    def _build_optimize_fn(self) -> Callable[[], tuple[Tensor, Tensor, float]]:
        """Build the function for pose and geometry optimization."""
        init_params = {k: v.value for k, v in self.get_parameters_by_prefix("dr.init").items()}
        zfar = self.get_parameter("dr.shader.zfar").value
        n_iter = self.get_parameter("dr.optim.n_iter").value
        rmat_init, tvec_init = renderer.look_at_view_transform(**init_params)
        is_ready = self._build_is_ready_fn()
        get_target = self._build_get_target_fn()

        def optimize_fn() -> tuple[Tensor, Tensor, float]:
            if not is_ready():
                return tvec_init[0], rmat_init[0], 0
            tvec, rmat, _, _ = self._optim.optimize(
                target=get_target(),
                n_iter=n_iter,
                logger=self._memory_logger,
                cameras=self._camera,
                zfar=zfar,
            )
            self._pose = (tvec, rmat)
            loss = self._memory_logger.flush()[n_iter]["loss"].min()
            return tvec, rmat, loss

        return optimize_fn

    def _build_is_ready_fn(self) -> Callable[[], bool]:
        """Build the function for checking if optimization is ready to run."""
        has_depth = "depth" in self.get_parameter("dr.shader.modalities").value

        def is_ready_silhouette() -> bool:
            return self._camera is not None and self._silhoutte is not None

        def is_ready_silhouette_depth() -> bool:
            return is_ready_silhouette() and self._depth is not None

        is_ready = is_ready_silhouette_depth if has_depth else is_ready_silhouette
        return is_ready

    def _build_get_target_fn(self) -> Callable[[], Tensor]:
        """Build the function for getting the target image according to the class configuration."""
        modalities = self.get_parameter("dr.shader.modalities").value
        zfar = self.get_parameter("dr.shader.zfar").value
        n_rep = self.get_parameter("dr.optim.n_rep").value

        def get_silhouette() -> Tensor:
            return self._crop_img(self._silhoutte)[None, :, :, None].expand([n_rep, -1, -1, -1])

        def get_depth() -> Tensor:
            clipped = torch.where(self._silhoutte > 0, self._depth, zfar)
            return self._crop_img(clipped)[None, :, :, None].expand([n_rep, -1, -1, -1])

        def get_silhouette_and_depth() -> Tensor:
            return torch.cat([get_silhouette(), get_depth()], dim=-1)

        if len(modalities) == 2:
            get_target = get_silhouette_and_depth
        else:
            get_target = get_silhouette if "silhouette" in modalities else get_depth
        return get_target

    def _crop_img(self, img: Tensor) -> Tensor:
        """Crop input image at the set center and size to prepare for rendering."""
        top = int(self._img_center[0] - self._size / 2)
        left = int(self._img_center[1] - self._size / 2)
        return VF.crop(img, top, left, self._size, self._size)

    def _reset(self) -> None:
        """Reset all attributes that are initialized from callbacks."""
        self._camera = None
        self._header = None
        self._silhoutte = None
        self._depth = None
        self._pose = None


def main(args=None):
    rclpy.init(args=args)
    tcp_calibration_p3d = TcpCalibrationP3d()
    rclpy.spin(tcp_calibration_p3d)
    tcp_calibration_p3d.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
