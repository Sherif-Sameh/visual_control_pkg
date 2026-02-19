#!/usr/bin/env python3

"""
Debugging node for the Apriltag detector from Isaac ROS Apriltag.
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from isaac_ros_apriltag_interfaces.msg import AprilTagDetection, AprilTagDetectionArray
from rclpy.node import Node
from sensor_msgs.msg import Image


class DetectorDebugger(Node):
    """Debugging node for April tag detector from Isaac ROS Apriltag.

    This node is used to publish messages based on the detector's outputs which could be used for
    debugging them. Given the AprilTagDetectionArray message from the detector, the debugger will
    publish a modified version of the input image with the tag's center, border adn ID overlayed
    on top.
    """

    def __init__(self):
        super().__init__("detector_debugger")

        # Initialize non-ROS class attributes
        self._bridge = CvBridge()
        self._img = None

        # Initialize ROS attributes
        self._pub_img = self.create_publisher(Image, "/detector_debugger/image", 0)
        self._sub_img = self.create_subscription(
            Image, "/image", self.callback_image, 0
        )
        self._sub_dtn = self.create_subscription(
            AprilTagDetectionArray, "/detections", self.callback_detection, 0
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
        # Constants
        RECT_COLOR, RECT_THICKNESS = (255, 0, 0), 2  # red and thickness in pixels
        CIRC_COLOR, CIRC_RADIUS = (0, 0, 255), 5  # blue and radius in pixels
        FONT_COLOR, FONT_THICKNESS = (0, 255, 0), 2  # green and thickness in pixels
        FONT_FACE, FONT_SCALE = cv2.FONT_HERSHEY_SIMPLEX, 1.0  # font type and size
        CENTER_OFFSET = (35, 35)

        # Format image
        detections: list[AprilTagDetection] = msg.detections  # for type inference
        for dtn in detections:
            # Draw circle at tag center
            self._img = cv2.circle(
                self._img,
                center=(int(dtn.center.x), int(dtn.center.y)),
                radius=CIRC_RADIUS,
                color=CIRC_COLOR,
                thickness=-1,  # filled
            )

            # Draw rectangle around tag boundary
            self._img = cv2.line(
                self._img,
                pt1=(int(dtn.corners[0].x), int(dtn.corners[0].y)),
                pt2=(int(dtn.corners[1].x), int(dtn.corners[1].y)),
                color=RECT_COLOR,
                thickness=RECT_THICKNESS,
            )
            self._img = cv2.line(
                self._img,
                pt1=(int(dtn.corners[1].x), int(dtn.corners[1].y)),
                pt2=(int(dtn.corners[2].x), int(dtn.corners[2].y)),
                color=RECT_COLOR,
                thickness=RECT_THICKNESS,
            )
            self._img = cv2.line(
                self._img,
                pt1=(int(dtn.corners[2].x), int(dtn.corners[2].y)),
                pt2=(int(dtn.corners[3].x), int(dtn.corners[3].y)),
                color=RECT_COLOR,
                thickness=RECT_THICKNESS,
            )
            self._img = cv2.line(
                self._img,
                pt1=(int(dtn.corners[3].x), int(dtn.corners[3].y)),
                pt2=(int(dtn.corners[0].x), int(dtn.corners[0].y)),
                color=RECT_COLOR,
                thickness=RECT_THICKNESS,
            )

            # Draw tag's ID on top of it
            self._img = cv2.putText(
                self._img,
                text=f"ID:{dtn.id}",
                org=(
                    int(dtn.center.x) - CENTER_OFFSET[0],
                    int(dtn.center.y) - CENTER_OFFSET[1],
                ),
                fontFace=FONT_FACE,
                fontScale=FONT_SCALE,
                color=FONT_COLOR,
                thickness=FONT_THICKNESS,
            )

        # Publish modified image
        msg = self._bridge.cv2_to_imgmsg(self._img, encoding="rgb8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub_img.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    detector_debugger = DetectorDebugger()
    rclpy.spin(detector_debugger)
    detector_debugger.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
