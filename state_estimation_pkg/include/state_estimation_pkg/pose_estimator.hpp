#ifndef POSE_ESTIMATOR
#define POSE_ESTIMATOR

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "isaac_ros_apriltag_interfaces/msg/april_tag_detection_array.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "std_msgs/msg/empty.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/transform_broadcaster.hpp"

#include "vc_core/actions/se3Features.hpp"
#include "vc_core/filters/ekf.hpp"
#include "vc_core/utils.hpp"

using isaac_ros_apriltag_interfaces::msg::AprilTagDetection;
using isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray;
using std::placeholders::_1;

class PoseEstimator : public rclcpp::Node
{
public:
    using State = se::EKF<manif::SE3d, se::ActionSE3Features<double>>::State;
    using Action = se::EKF<manif::SE3d, se::ActionSE3Features<double>>::Action;
    using Measurement = se::EKF<manif::SE3d, se::ActionSE3Features<double>>::Measurement;
    using Covariance = se::EKF<manif::SE3d, se::ActionSE3Features<double>>::Covariance;

public:
    PoseEstimator(const rclcpp::NodeOptions &options);

private:
    void publish_pose(const std_msgs::msg::Header &header);
    void make_pose_tf(const std_msgs::msg::Header &header);
    void callback_cam_twist(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
    void callback_pose(const AprilTagDetectionArray::SharedPtr msg);
    void callback_rst(const std_msgs::msg::Empty::SharedPtr msg);
    rcl_interfaces::msg::SetParametersResult
    callback_params(const std::vector<rclcpp::Parameter> &parameters);

    void init_ekf();

private:
    // General Attributes
    bool m_pub_pred;
    std::string m_pose_frame;
    std::pair<double, double> m_pose_P_thr;
    utils::structs::PeriodEMACalculator<double> m_twist_cb_pc;

    // EKF Attributes
    bool m_ekf_init;
    se::EKF<manif::SE3d, se::ActionSE3Features<double>> m_ekf;
    static constexpr se::ActionSE3Features<double> action_fn = se::ActionSE3Features<double>();

    // ROS Attributes
    rclcpp::Publisher<AprilTagDetectionArray>::SharedPtr m_pub_pose{nullptr};
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr m_sub_cam_twist{nullptr};
    rclcpp::Subscription<AprilTagDetectionArray>::SharedPtr m_sub_pose{nullptr};
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr m_sub_rst{nullptr};
    std::unique_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster{nullptr};
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr m_cbh_param{nullptr};
};

#endif