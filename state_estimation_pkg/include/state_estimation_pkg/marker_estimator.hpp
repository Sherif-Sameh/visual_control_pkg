#ifndef MARKER_ESTIMATOR
#define MARKER_ESTIMATOR

#include <array>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "isaac_ros_apriltag_interfaces/msg/april_tag_detection_array.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/transform_broadcaster.hpp"

#include "vc_core/actions/se3Features.hpp"
#include "vc_core/filters/ekf.hpp"
#include "vc_core/utils/geometry.hpp"
#include "vc_core/utils/mappings.hpp"
#include "vc_core/utils/structs.hpp"

using isaac_ros_apriltag_interfaces::msg::AprilTagDetection;
using isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray;
using std::placeholders::_1;

class MarkerEstimator : public rclcpp::Node
{
public:
    using ManifEKFStamped =
        utils::structs::AnyStamped<se::EKF<manif::SE3d, se::ActionSE3Features<double>>>;

    using State = se::EKF<manif::SE3d, se::ActionSE3Features<double>>::State;
    using Action = se::EKF<manif::SE3d, se::ActionSE3Features<double>>::Action;
    using Measurement = se::EKF<manif::SE3d, se::ActionSE3Features<double>>::Measurement;
    using Covariance = se::EKF<manif::SE3d, se::ActionSE3Features<double>>::Covariance;

public:
    MarkerEstimator(const rclcpp::NodeOptions &options);

private:
    void publish_tag(const std_msgs::msg::Header &header, const std::string &tag_family);
    void make_tag_tfs(const std_msgs::msg::Header &header, const std::string &tag_family);
    void callback_cam_info(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void callback_cam_twist(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
    void callback_tag(const AprilTagDetectionArray::SharedPtr msg);
    rcl_interfaces::msg::SetParametersResult
    callback_params(const std::vector<rclcpp::Parameter> &parameters);

    void init_ekf();
    AprilTagDetection create_tag_detection(const std::string &family, const int id,
                                           const Eigen::Isometry3d &T_tag_cam,
                                           const Covariance &cov);
    std::array<Eigen::Vector2d, 4> project_points(const Eigen::Isometry3d &T_tag_cam);

private:
    // General Attributes
    double m_tag_timeout;
    std::pair<double, double> m_tag_P_thr;
    utils::structs::PeriodEMACalculator<double> m_tag_cb_pc;
    std::optional<Eigen::Matrix3d> m_cam_K;
    std::array<Eigen::Vector3d, 4> m_tag_pts;

    // EKF Attributes
    Covariance m_ekf_P0, m_ekf_Q, m_ekf_R;
    std::unordered_map<int, ManifEKFStamped> m_ekf_map;
    static constexpr se::ActionSE3Features<double> action_fn = se::ActionSE3Features<double>();

    // ROS Attributes
    rclcpp::Publisher<AprilTagDetectionArray>::SharedPtr m_pub_tag{nullptr};
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr m_sub_cam_info{nullptr};
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr m_sub_cam_twist{nullptr};
    rclcpp::Subscription<AprilTagDetectionArray>::SharedPtr m_sub_tag{nullptr};
    std::unique_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster{nullptr};
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr m_cbh_param{nullptr};
};

#endif