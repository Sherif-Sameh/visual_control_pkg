#ifndef IBVS_CONTROLLER
#define IBVS_CONTROLLER

#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <visp3/core/vpCameraParameters.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/vs/vpServo.h>

#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "isaac_ros_apriltag_interfaces/msg/april_tag_detection_array.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "tf2_ros/buffer.hpp"
#include "tf2_ros/transform_listener.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"

#include "vc_core/robot/vpRobotRos.hpp"
#include "vc_core/utils.hpp"

using isaac_ros_apriltag_interfaces::msg::AprilTagDetection;
using isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray;
using std::placeholders::_1;
using trajectory_msgs::msg::MultiDOFJointTrajectory;
using vc::visp::vpRobotRos;
using namespace std::chrono_literals;

class IbvsController : public rclcpp::Node
{
public:
    IbvsController();
    ~IbvsController();

private:
    void publish_traj(const std::vector<double> &qdot);
    void publish_perr(const std::vector<double> &perr);
    void publish_cam_twist(const vpColVector &v_c);
    void callback_timer();
    void callback_js(const sensor_msgs::msg::JointState::SharedPtr msg);
    void callback_cam_info(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void callback_tag(const AprilTagDetectionArray::SharedPtr msg);
    void callback_traj_des(const MultiDOFJointTrajectory::SharedPtr msg);
    rcl_interfaces::msg::SetParametersResult
    callback_params(const std::vector<rclcpp::Parameter> &parameters);

    void init_robot();
    void init_controller();
    bool update_features(const std::vector<AprilTagDetection> &detections);
    std::vector<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>::const_iterator
    find_traj_point(const rclcpp::Duration &elapsed);
    bool has_converged();

private:
    // General Attributes
    int m_tag_id;
    double m_timeout;
    std::string m_base_frame;
    std::string m_ee_frame;
    std::string m_cam_frame;
    std::vector<std::string> m_joint_names;

    // Controller Attributes
    double m_conv_eps;
    rclcpp::Time m_ctrl_ts, m_traj_ts;
    std::array<vpPoint, 4> m_points;
    std::array<vpFeaturePoint, 4> m_p, m_pd;
    MultiDOFJointTrajectory::SharedPtr m_traj_msg{nullptr};
    std::optional<vpCameraParameters> m_cam_params;
    utils::structs::EMA<vpColVector, double> m_v_cam_ff;
    vpColVector m_lambda;
    vpRobotRos m_robot;
    vpServo m_controller;

    // ROS Attributes
    rclcpp::TimerBase::SharedPtr m_timer;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr m_pub_traj{nullptr};
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr m_pub_perr{nullptr};
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr m_pub_cam_twist{nullptr};
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr m_sub_js{nullptr};
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr m_sub_cam_info{nullptr};
    rclcpp::Subscription<AprilTagDetectionArray>::SharedPtr m_sub_tag{nullptr};
    rclcpp::Subscription<MultiDOFJointTrajectory>::SharedPtr m_sub_traj_des{nullptr};
    std::shared_ptr<tf2_ros::TransformListener> m_tf_listener{nullptr};
    std::unique_ptr<tf2_ros::Buffer> m_tf_buffer;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr m_cbh_param{nullptr};
};

#endif