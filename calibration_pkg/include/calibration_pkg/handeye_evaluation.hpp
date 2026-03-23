#ifndef HANDEYE_EVALUATION_NODE
#define HANDEYE_EVALUATION_NODE

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <Eigen/Geometry>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_apriltag_interfaces/msg/april_tag_detection_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.hpp"
#include "tf2_ros/transform_listener.h"

#include "vc_core/calibration/handeyeEngine.hpp"
#include "vc_core/filters/lowPassFilter.hpp"
#include "vc_core/utils.hpp"

using isaac_ros_apriltag_interfaces::msg::AprilTagDetection;
using isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray;
using std::placeholders::_1;

class HandeyeEvaluation : public rclcpp::Node
{
public:
    HandeyeEvaluation();

private:
    void publish_target();
    void publish_target_tf(const std_msgs::msg::Header &header);
    void publish_error(const Eigen::Isometry3d &cam_target);
    void callback_dtn(const AprilTagDetectionArray::SharedPtr msg);

    void init_calibration_engine();
    void update_base_target(const Eigen::Isometry3d &base_ee, const Eigen::Isometry3d &cam_target);
    bool has_converged(const Eigen::Isometry3d &base_ee);

private:
    // General Attributes
    std::string m_base_frame;
    std::string m_ee_frame;

    // Calibration Evaluation Attributes
    std::size_t m_n_poses;
    Eigen::Isometry3d m_ee_cam, m_base_ee_d, m_target_cam_d;
    std::optional<se::LowPassFilter<manif::SE3d>> m_base_target_lpf;
    utils::structs::PoseTolerance<double> m_conv_tol;
    calib::HandeyeEngine<double> m_engine;

    // ROS Attributes
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_pub_target{nullptr};
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr m_pub_error{nullptr};
    rclcpp::Subscription<AprilTagDetectionArray>::SharedPtr m_sub_dtn{nullptr};
    std::shared_ptr<tf2_ros::TransformListener> m_tf_listener{nullptr};
    std::unique_ptr<tf2_ros::Buffer> m_tf_buffer{nullptr};
    std::unique_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster{nullptr};
};

#endif