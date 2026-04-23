#ifndef HANDEYE_CALIBRATION_NODE
#define HANDEYE_CALIBRATION_NODE

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <Eigen/Geometry>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_apriltag_interfaces/msg/april_tag_detection_array.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/empty.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.hpp"
#include "tf2_ros/transform_listener.h"

#include "vc_core/calibration/handeyeEngine.hpp"
#include "vc_core/utils.hpp"

using isaac_ros_apriltag_interfaces::msg::AprilTagDetection;
using isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray;
using std::placeholders::_1;

class HandeyeCalibration : public rclcpp::Node
{
public:
    HandeyeCalibration();

private:
    void publish_target();
    void publish_target_tf(const std_msgs::msg::Header &header);
    void publish_error(const Eigen::Isometry3d &ee_cam);
    void callback_dtn(const AprilTagDetectionArray::SharedPtr msg);
    void callback_rst(const std_msgs::msg::Empty::SharedPtr msg);
    rcl_interfaces::msg::SetParametersResult
    callback_params(const std::vector<rclcpp::Parameter> &parameters);

    void init_calibration_engine();
    bool has_converged(const Eigen::Isometry3d &cam_target);

private:
    // General Attributes
    std::string m_base_frame;
    std::string m_ee_frame;
    std::string m_cam_frame;

    // Calibration Attributes
    bool m_done, m_target_pub;
    std::string m_path;
    Eigen::Isometry3d m_target_cam;
    std::optional<Eigen::Isometry3d> m_pose_gt;
    utils::structs::PoseTolerance<double> m_conv_tol;
    calib::HandeyeEngine<double> m_engine;

    // ROS Attributes
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_pub_target{nullptr};
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_pub_error{nullptr};
    rclcpp::Subscription<AprilTagDetectionArray>::SharedPtr m_sub_dtn{nullptr};
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_sub_rst{nullptr};
    std::shared_ptr<tf2_ros::TransformListener> m_tf_listener{nullptr};
    std::unique_ptr<tf2_ros::Buffer> m_tf_buffer{nullptr};
    std::unique_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster{nullptr};
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr m_cbh_param{nullptr};
};

#endif