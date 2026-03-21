#ifndef CHARUCO_DETECTOR
#define CHARUCO_DETECTOR

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_apriltag_interfaces/msg/april_tag_detection_array.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/transform_broadcaster.hpp"

#include "vc_core/utils/geometry.hpp"
#include "vc_core/utils/str2enum.hpp"

using isaac_ros_apriltag_interfaces::msg::AprilTagDetection;
using isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray;
using std::placeholders::_1;

class CharucoDetector : public rclcpp::Node
{
public:
    CharucoDetector(const rclcpp::NodeOptions &options);

private:
    void publish_dtn(const std_msgs::msg::Header &header, const Eigen::Isometry3d &pose);
    void publish_dtn_tf(const std_msgs::msg::Header &header, const Eigen::Isometry3d &pose);
    void callback_cam_info(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void callback_image(const sensor_msgs::msg::Image::SharedPtr msg);
    rcl_interfaces::msg::SetParametersResult
    callback_params(const std::vector<rclcpp::Parameter> &parameters);

    void init_detector_params();
    Eigen::Isometry3d transform_board_pose(const cv::Vec3d &tvec, const cv::Vec3d &rvec);

private:
    // General Attributes
    bool m_visualize;
    Eigen::Vector2d m_board_size;
    std::optional<cv::Mat> m_cam_K;

    // Detector Attributes
    cv::Ptr<cv::aruco::Dictionary> m_dict{nullptr};
    cv::Ptr<cv::aruco::CharucoBoard> m_board{nullptr};
    cv::Ptr<cv::aruco::DetectorParameters> m_params{nullptr};

    // ROS Attributes
    rclcpp::Publisher<AprilTagDetectionArray>::SharedPtr m_pub_dtn{nullptr};
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_pub_image{nullptr};
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr m_sub_cam_info{nullptr};
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_sub_image{nullptr};
    std::unique_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster{nullptr};
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr m_cbh_param{nullptr};
};

#endif