#ifndef PBVS_CONTROLLER
#define PBVS_CONTROLLER

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <visp3/visual_features/vpFeatureThetaU.h>
#include <visp3/visual_features/vpFeatureTranslation.h>
#include <visp3/vs/vpServo.h>

#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "isaac_ros_apriltag_interfaces/msg/april_tag_detection.hpp"
#include "isaac_ros_apriltag_interfaces/msg/april_tag_detection_array.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "tf2/exceptions.hpp"
#include "tf2_ros/buffer.hpp"
#include "tf2_ros/transform_listener.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

#include "utils/conversions/geometry.hpp"
#include "utils/conversions/mappings.hpp"
#include "vpRobotRos.hpp"

using isaac_ros_apriltag_interfaces::msg::AprilTagDetection;
using isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray;
using std::placeholders::_1;
using vc::visp::vpRobotRos;
using namespace std::chrono_literals;

class PbvsController : public rclcpp::Node
{
public:
    PbvsController();
    ~PbvsController();

private:
    void post_init();
    void publish_traj(const std::vector<double> &qdot);
    void publish_perr(const std::vector<double> &perr);
    void callback_js(const sensor_msgs::msg::JointState::SharedPtr msg);
    void callback_tag(const AprilTagDetectionArray::SharedPtr msg);
    rcl_interfaces::msg::SetParametersResult
    callback_params(const std::vector<rclcpp::Parameter> &parameters);
    void init_robot();
    void init_controller();
    void update_desired_features();
    void update_features(const std::vector<AprilTagDetection> &detections,
                         std::vector<int> &valid_ids, std::vector<int> &invalid_ids);
    bool has_converged(const std::vector<int> &valid_ids);
    bool lookup_transform(const std::string &target_frame, const std::string &source_frame,
                          vpHomogeneousMatrix &t);

private:
    // General Attributes
    std::string m_base_frame;
    std::string m_ee_frame;
    std::string m_cam_frame;
    std::string m_tag_family;
    std::vector<int> m_tag_ids;
    std::vector<std::string> m_joint_names;

    // ViSP Attributes
    std::unordered_map<int, vpFeatureTranslation> m_t;
    std::unordered_map<int, vpFeatureThetaU> m_tu;
    std::unordered_map<int, vpHomogeneousMatrix> m_cdMo;
    std::pair<double, double> m_conv_eps;
    vpColVector m_lambda;
    vpRobotRos m_robot;
    vpServo m_controller;

    // ROS Attributes
    rclcpp::TimerBase::SharedPtr m_timer_setup;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr m_pub_traj{nullptr};
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr m_pub_perr{nullptr};
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr m_sub_js{nullptr};
    rclcpp::Subscription<AprilTagDetectionArray>::SharedPtr m_sub_tag{nullptr};
    std::shared_ptr<tf2_ros::TransformListener> m_tf_listener{nullptr};
    std::unique_ptr<tf2_ros::Buffer> m_tf_buffer;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr m_cbh_param;
};

#endif