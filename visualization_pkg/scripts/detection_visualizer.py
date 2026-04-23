#!/usr/bin/env python3

"""
ROS node for visualizing Apriltag detections from the Isaac ROS Apriltag detector.
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from isaac_ros_apriltag_interfaces.msg import AprilTagDetection, AprilTagDetectionArray
from rclpy.node import Node
from sensor_msgs.msg import Image


class DetectionVisualizer(Node):
    """ROS node for visualizing Apriltag detections from the Isaac ROS Apriltag detector.

    This node listens to AprilTag detection messages published by the detector as well as the raw
    RGB images used for detection. Given an AprilTagDetectionArray message from the detector and
    its corresponding Image message, it will publish a modified version of the image with the tag's
    center, border adn ID overlayed on top.
    """

    CENTER_COLOR = (0, 0, 255)  # blue
    BORDER_COLOR = (255, 0, 0)  # red
    FONT_COLOR = (0, 255, 0)  # green
    FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        super().__init__("detection_visualizer")

        # Declare ROS parameters
        self.declare_parameter("center.radius", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("border.thickness", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("font.thickness", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("font.scale", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("font.offset_x", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("font.offset_y", rclpy.Parameter.Type.INTEGER)

        # Initialize non-ROS class attributes
        self._center_radius = self.get_parameter("center.radius").value
        self._border_thickness = self.get_parameter("border.thickness").value
        self._font_thickness = self.get_parameter("font.thickness").value
        self._font_scale = self.get_parameter("font.scale").value
        self._font_offset_x = self.get_parameter("font.offset_x").value
        self._font_offset_y = self.get_parameter("font.offset_y").value
        self._bridge = CvBridge()
        self._img = None

        # Initialize ROS attributes
        self._pub_img = self.create_publisher(Image, "/detection_visualizer/image", 1)
        self._sub_img = self.create_subscription(Image, "/image", self.callback_image, 1)
        self._sub_dtn = self.create_subscription(
            AprilTagDetectionArray, "/detections", self.callback_detection, 1
        )

    def callback_image(self, msg: Image) -> None:
        """Callback function for input image.

        Stores input image for use in detection callback.

        Args:
            msg: sensor_msgs/Image input messsage.
        """
        self._img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def callback_detection(self, msg: AprilTagDetectionArray) -> None:
        """Callback function for the Apriltag detections.

        Publishes modified version of stored input image with tags overlayed on top of it.

        Args:
            msg: AprilTagDetectionArray message containing the relative poses of each tag.
        """
        # Publish modified image if input image is available
        if self._img is not None:
            self._publish_tag_image(msg)
            self._img = None  # reset stored image after publishing

    def _publish_tag_image(self, msg: AprilTagDetectionArray) -> None:
        """Publish stored input image with tags overlayed on top.

        Args:
            msg: AprilTagDetectionArray message containing the relative poses of each tag.
        """
        detections: list[AprilTagDetection] = msg.detections  # for type inference
        for dtn in detections:
            # Draw circle at tag center
            self._img = cv2.circle(
                self._img,
                center=(int(dtn.center.x), int(dtn.center.y)),
                radius=self._center_radius,
                color=self.CENTER_COLOR,
                thickness=-1,  # filled
            )

            # Draw rectangle around tag boundary
            self._img = cv2.line(
                self._img,
                pt1=(int(dtn.corners[0].x), int(dtn.corners[0].y)),
                pt2=(int(dtn.corners[1].x), int(dtn.corners[1].y)),
                color=self.BORDER_COLOR,
                thickness=self._border_thickness,
            )
            self._img = cv2.line(
                self._img,
                pt1=(int(dtn.corners[1].x), int(dtn.corners[1].y)),
                pt2=(int(dtn.corners[2].x), int(dtn.corners[2].y)),
                color=self.BORDER_COLOR,
                thickness=self._border_thickness,
            )
            self._img = cv2.line(
                self._img,
                pt1=(int(dtn.corners[2].x), int(dtn.corners[2].y)),
                pt2=(int(dtn.corners[3].x), int(dtn.corners[3].y)),
                color=self.BORDER_COLOR,
                thickness=self._border_thickness,
            )
            self._img = cv2.line(
                self._img,
                pt1=(int(dtn.corners[3].x), int(dtn.corners[3].y)),
                pt2=(int(dtn.corners[0].x), int(dtn.corners[0].y)),
                color=self.BORDER_COLOR,
                thickness=self._border_thickness,
            )

            # Draw tag's ID on top of it
            self._img = cv2.putText(
                self._img,
                text=f"ID:{dtn.id}",
                org=(
                    int(dtn.center.x) + self._font_offset_x,
                    int(dtn.center.y) + self._font_offset_y,
                ),
                fontFace=self.FONT_TYPE,
                fontScale=self._font_scale,
                color=self.FONT_COLOR,
                thickness=self._font_thickness,
            )

        # Publish modified image
        msg = self._bridge.cv2_to_imgmsg(self._img, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub_img.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    detection_visualizer = DetectionVisualizer()
    rclpy.spin(detection_visualizer)
    detection_visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
