#include "calibration_pkg/handeye_evaluation.hpp"

HandeyeEvaluation::HandeyeEvaluation() : Node("handeye_evaluation")
{
    // Declare ROS parameters
    this->declare_parameter("frame.base_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.ee_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("calib.pose_est", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("calib.conv_ttol", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("calib.conv_rtol", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("calib.lpf_coeff", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("calib.eng.n_poses", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("calib.eng.rot_angle", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("calib.eng.dist_to_target", rclcpp::PARAMETER_DOUBLE);

    // Initialize non-ROS class attributes
    m_base_frame = this->get_parameter("frame.base_frame").as_string();
    m_ee_frame = this->get_parameter("frame.ee_frame").as_string();
    auto pose_est = this->get_parameter("calib.pose_est").as_double_array();
    if (!(pose_est.size() == 7))
    {
        RCLCPP_ERROR_STREAM(this->get_logger(), "Pose vector must be of size 7. Got "
                                                    << pose_est.size() << " instead.");
        rclcpp::shutdown();
    }
    Eigen::Translation3d t(pose_est[0], pose_est[1], pose_est[2]);
    Eigen::Quaterniond q(pose_est[3], pose_est[4], pose_est[5], pose_est[6]);
    m_ee_cam = t * q;
    m_conv_tol.m_ttol = this->get_parameter("calib.conv_ttol").as_double();
    m_conv_tol.m_rtol = this->get_parameter("calib.conv_rtol").as_double();
    init_calibration_engine();

    // Initialize ROS attributes
    m_pub_target = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/handeye_evaluation/command", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    m_pub_error = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/handeye_evaluation/pose_error", 10);
    m_sub_dtn = this->create_subscription<AprilTagDetectionArray>(
        "/detections", 1, std::bind(&HandeyeEvaluation::callback_dtn, this, _1));
    m_tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    m_tf_listener = std::make_shared<tf2_ros::TransformListener>(*m_tf_buffer);
    m_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
}

void HandeyeEvaluation::publish_target()
{
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = this->get_clock()->now();
    msg.pose = tf2::toMsg(m_base_ee_d);
    m_pub_target->publish(msg);
}

void HandeyeEvaluation::publish_target_tf(const std_msgs::msg::Header &header)
{
    geometry_msgs::msg::TransformStamped transform = tf2::eigenToTransform(m_base_ee_d);
    transform.header.stamp = header.stamp;
    transform.header.frame_id = m_base_frame;
    transform.child_frame_id = m_ee_frame + ":0";
    m_tf_broadcaster->sendTransform(transform);
}

void HandeyeEvaluation::publish_error(const Eigen::Isometry3d &cam_target)
{
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = this->get_clock()->now();

    Eigen::Isometry3d error = cam_target * m_target_cam_d;
    msg.pose = tf2::toMsg(error);
    m_pub_error->publish(msg);
}

void HandeyeEvaluation::callback_dtn(const AprilTagDetectionArray::SharedPtr msg)
{
    std::size_t n = m_engine.getNPosesStored();
    if (n == m_n_poses) return;

    // Get target wrt camera pose from detections
    auto it = std::find_if(msg->detections.cbegin(), msg->detections.cend(),
                           [](const AprilTagDetection &dtn) { return dtn.id == 0; });
    if (it == msg->detections.cend()) return; // invalid detection
    Eigen::Isometry3d cam_target;
    tf2::fromMsg((*it).pose.pose.pose, cam_target);

    // Get ee wrt base pose and initialize target wrt base pose
    Eigen::Isometry3d base_ee;
    if (!utils::ros_tf2::lookup_transform(m_base_frame, m_ee_frame, m_tf_buffer, base_ee)) return;
    if (!m_base_target_lpf.has_value()) update_base_target(base_ee, cam_target);

    if (has_converged(base_ee))
    {
        // Update target wrt base pose/publish pose error
        if (n < m_n_poses / 2)
        {
            update_base_target(base_ee, cam_target);
        }
        else
        {
            publish_error(cam_target);
        }
        // Add poses to increment count and update setpoint
        m_engine.addPoses(base_ee, cam_target);
        bool updated = m_engine.getNextCameraPose(m_target_cam_d);
        if (updated)
        {
            Eigen::Isometry3d base_target = (*m_base_target_lpf).getState().isometry();
            m_base_ee_d = base_target * m_target_cam_d * m_ee_cam.inverse();
        }
    }

    // Publish latest target
    publish_target();
    publish_target_tf(msg->header);
}

void HandeyeEvaluation::init_calibration_engine()
{
    m_n_poses = static_cast<std::size_t>(this->get_parameter("calib.eng.n_poses").as_int());
    double rot_angle = this->get_parameter("calib.eng.rot_angle").as_double();
    double dist_to_target = this->get_parameter("calib.eng.dist_to_target").as_double();

    m_engine.setNPoses(m_n_poses, false);
    m_engine.setRotAngle(rot_angle, false);
    m_engine.setDistToTarget(dist_to_target, true);
    m_engine.getNextCameraPose(m_target_cam_d);
}

void HandeyeEvaluation::update_base_target(const Eigen::Isometry3d &base_ee,
                                           const Eigen::Isometry3d &cam_target)
{
    Eigen::Isometry3d base_target = base_ee * m_ee_cam * cam_target;
    if (m_base_target_lpf.has_value())
    {
        (*m_base_target_lpf).update(base_target);
    }
    else // late initialization
    {
        m_base_target_lpf.emplace(this->get_parameter("calib.lpf_coeff").as_double());
        (*m_base_target_lpf).setState(base_target);
        m_base_ee_d = base_target * m_target_cam_d * m_ee_cam.inverse();
    }
}

bool HandeyeEvaluation::has_converged(const Eigen::Isometry3d &base_ee)
{
    Eigen::Isometry3d error = m_base_ee_d.inverse() * base_ee;
    double terror = error.translation().norm();
    double rerror = Eigen::AngleAxisd(error.rotation()).angle();
    return terror < m_conv_tol.m_ttol && rerror < m_conv_tol.m_rtol;
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto handeye_evaluation = std::make_shared<HandeyeEvaluation>();
    rclcpp::spin(handeye_evaluation);
    rclcpp::shutdown();
    return 0;
}
