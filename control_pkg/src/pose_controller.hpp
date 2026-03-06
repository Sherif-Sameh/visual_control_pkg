#ifndef POSE_CONTROLLER
#define POSE_CONTROLLER

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <visp3/core/vpColVector.h>
#include <visp3/core/vpHomogeneousMatrix.h>

#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
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

using std::placeholders::_1;
using vc::visp::vpRobotRos;
using namespace std::chrono_literals;

class PoseController : public rclcpp::Node
{
public:
    PoseController();
    ~PoseController();

private:
    void publish_perr(const vpPoseVector &edPe);
    void publish_traj(const std::vector<double> &qdot);
    void callback_gp(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void callback_js(const sensor_msgs::msg::JointState::SharedPtr msg);
    rcl_interfaces::msg::SetParametersResult
    callback_params(const std::vector<rclcpp::Parameter> &parameters);
    void init_robot();
    void init_controller();
    bool has_converged(const vpPoseVector &edPe);
    bool lookup_transform(const std::string &target_frame, const std::string &source_frame,
                          vpHomogeneousMatrix &t);

private:
    // General Attributes
    std::string m_base_frame;
    std::string m_ee_frame;
    std::vector<std::string> m_joint_names;

    // Controller Attributes
    std::optional<vpHomogeneousMatrix> m_fMed;
    std::pair<double, double> m_conv_eps;
    vpColVector m_lambda;
    vpRobotRos m_robot;

    // ROS Attributes
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr m_pub_perr{nullptr};
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr m_pub_traj{nullptr};
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr m_sub_gp{nullptr};
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr m_sub_js{nullptr};
    std::shared_ptr<tf2_ros::TransformListener> m_tf_listener{nullptr};
    std::unique_ptr<tf2_ros::Buffer> m_tf_buffer;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr m_cbh_param;
};

#endif