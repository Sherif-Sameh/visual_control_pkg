#include "calibration_pkg/handeye_calibration.hpp"

HandeyeCalibration::HandeyeCalibration() : Node("handeye_calibration")
{
    // Declare ROS parameters
    this->declare_parameter("base_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("ee_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("cam_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("config_path", rclcpp::PARAMETER_STRING);
    this->declare_parameter("pose_gt", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("calib.conv_ttol", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("calib.conv_rtol", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("calib.eng.n_poses", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("calib.eng.dist_to_target", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("calib.eng.calib_method", rclcpp::PARAMETER_STRING);

    // Initialize non-ROS class attributes
    m_base_frame = this->get_parameter("base_frame").as_string();
    m_ee_frame = this->get_parameter("ee_frame").as_string();
    m_cam_frame = this->get_parameter("cam_frame").as_string();
    m_config_path = this->get_parameter("config_path").as_string();
    auto pose_gt = this->get_parameter("pose_gt").as_double_array();
    if (pose_gt.size() == 7)
    {
        Eigen::Translation3d t(pose_gt[0], pose_gt[1], pose_gt[2]);
        Eigen::Quaterniond q(pose_gt[3], pose_gt[4], pose_gt[5], pose_gt[6]);
        m_pose_gt = t * q;
    }
    m_calib_done = false;
    m_conv_tol.m_ttol = this->get_parameter("calib.conv_ttol").as_double();
    m_conv_tol.m_rtol = this->get_parameter("calib.conv_rtol").as_double();
    init_calibration_engine();

    // Initialize ROS attributes
    m_pub_target =
        this->create_publisher<MultiDOFJointTrajectory>("/handeye_calibration/command", 0);
    m_pub_error = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/handeye_calibration/pose_error", 10);
    m_sub_dtn = this->create_subscription<AprilTagDetectionArray>(
        "/detections", 0, std::bind(&HandeyeCalibration::callback_dtn, this, _1));
    m_sub_rst = this->create_subscription<std_msgs::msg::Empty>(
        "/handeye_calibration/restart", 0, std::bind(&HandeyeCalibration::callback_rst, this, _1));
    m_tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    m_tf_listener = std::make_shared<tf2_ros::TransformListener>(*m_tf_buffer);
    m_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    m_cbh_param = this->add_on_set_parameters_callback(
        std::bind(&HandeyeCalibration::callback_params, this, _1));
}

void HandeyeCalibration::post_init()
{
    publish_target();
    publish_target_tf();
}

void HandeyeCalibration::publish_target()
{
    MultiDOFJointTrajectory msg;
    msg.header.stamp = this->get_clock()->now();
    msg.joint_names.push_back("0");

    trajectory_msgs::msg::MultiDOFJointTrajectoryPoint cmd;
    cmd.transforms.push_back(tf2::eigenToTransform(m_target_cam).transform);
    cmd.velocities.push_back(geometry_msgs::msg::Twist());
    msg.points.push_back(cmd);
    m_pub_target->publish(msg);
}

void HandeyeCalibration::publish_target_tf()
{
    geometry_msgs::msg::TransformStamped transform = tf2::eigenToTransform(m_target_cam);
    transform.header.stamp = this->get_clock()->now();
    transform.header.frame_id = "charuco:0f";
    transform.child_frame_id = m_cam_frame + ":0";
    m_tf_broadcaster->sendTransform(transform);
}

void HandeyeCalibration::publish_error(const Eigen::Isometry3d &ee_cam)
{
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = this->get_clock()->now();

    Eigen::Isometry3d error = (*m_pose_gt).inverse() * ee_cam;
    msg.pose = tf2::toMsg(error);
    m_pub_error->publish(msg);
}

void HandeyeCalibration::callback_dtn(const AprilTagDetectionArray::SharedPtr msg)
{
    if (m_calib_done) return;

    // Get target wrt camera pose from detections
    auto it = std::find_if(msg->detections.cbegin(), msg->detections.cend(),
                           [](const AprilTagDetection &dtn) { return dtn.id == 0; });
    if (it == msg->detections.cend()) return; // invalid detection
    Eigen::Isometry3d cam_target;
    tf2::fromMsg((*it).pose.pose.pose, cam_target);
    if (!has_converged(cam_target)) return; // not yet converged to setpoint

    // Store poses ee wrt base and target wrt camera poses
    Eigen::Isometry3d base_ee;
    if (!utils::ros_tf2::lookup_transform(m_base_frame, m_ee_frame, m_tf_buffer, base_ee)) return;
    m_engine.addPoses(base_ee, cam_target);

    // Update setpoint or solve for params if done
    bool updated = m_engine.getNextCameraPose(m_target_cam);
    if (updated)
    {
        publish_target();
        publish_target_tf();
    }
    else // all poses collected
    {
        m_calib_done = true;
        Eigen::Isometry3d ee_cam;
        m_engine.calibrateHandEye(m_config_path, ee_cam);
        if (m_pose_gt.has_value())
        {
            publish_error(ee_cam);
        }
    }
}

void HandeyeCalibration::callback_rst(const std_msgs::msg::Empty::SharedPtr msg)
{
    (void)msg;
    // reset calibration engine and publish first setpoint
    m_calib_done = false;
    init_calibration_engine();
    post_init();
}

rcl_interfaces::msg::SetParametersResult
HandeyeCalibration::callback_params(const std::vector<rclcpp::Parameter> &parameters)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";
    auto failed = [&result](const std::string &reason)
    {
        result.successful = false;
        result.reason = reason;
    };

    // Check for params that are allowed to change at runtime
    for (const auto &param : parameters)
    {
        if (param.get_name().find("calib.eng") == std::string::npos)
        {
            failed(param.get_name() + " cannot be changed at runtime.");
            break;
        }
        if (param.get_name() == "calib.eng.n_poses" &&
            param.get_type() != rclcpp::PARAMETER_INTEGER)
        {
            failed("calib.eng.n_poses must be integer");
            break;
        }
        if (param.get_name() == "calib.eng.dist_to_target" &&
            param.get_type() != rclcpp::PARAMETER_DOUBLE)
        {
            failed("calib.eng.dist_to_target must be double");
            break;
        }
        if (param.get_name() == "calib.eng.calib_method" &&
            param.get_type() != rclcpp::PARAMETER_STRING)
        {
            failed("calib.eng.calib_method must be string");
            break;
        }
    }
    if (result.successful)
    {
        init_calibration_engine();
    }
    return result;
}

void HandeyeCalibration::init_calibration_engine()
{
    std::size_t n_poses =
        static_cast<std::size_t>(this->get_parameter("calib.eng.n_poses").as_int());
    double dist_to_target = this->get_parameter("calib.eng.dist_to_target").as_double();
    cv::HandEyeCalibrationMethod calib_method = utils::str2enum::cvHandEyeCalibrationMethodMap.at(
        this->get_parameter("calib.eng.calib_method").as_string());

    m_engine.setNPoses(n_poses, false);
    m_engine.setDistToTarget(dist_to_target, true);
    m_engine.setCalibMethod(calib_method);
    m_engine.getNextCameraPose(m_target_cam);
}

bool HandeyeCalibration::has_converged(const Eigen::Isometry3d &cam_target)
{
    Eigen::Isometry3d error = cam_target * m_target_cam;
    double terror = error.translation().norm();
    double rerror = Eigen::AngleAxisd(error.rotation()).angle();
    return terror < m_conv_tol.m_ttol && rerror < m_conv_tol.m_rtol;
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto handeye_calibration = std::make_shared<HandeyeCalibration>();
    handeye_calibration->post_init();
    rclcpp::spin(handeye_calibration);
    rclcpp::shutdown();
    return 0;
}
