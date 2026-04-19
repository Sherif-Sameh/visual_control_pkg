#include "state_estimation_pkg/pose_estimator.hpp"

PoseEstimator::PoseEstimator(const rclcpp::NodeOptions &options) : Node("pose_estimator", options)
{
    // Declare ROS parameters
    this->declare_parameter("pose.frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("pose.P_tthr", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("pose.P_rthr", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("pose.ema_alpha", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ekf.P0_diag", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("ekf.Q_diag", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("ekf.R_diag", rclcpp::PARAMETER_DOUBLE_ARRAY);

    // Initialize non-ROS class attributes
    m_pose_frame = this->get_parameter("pose.frame").as_string();
    m_pose_P_thr.first = this->get_parameter("pose.P_tthr").as_double();
    m_pose_P_thr.second = this->get_parameter("pose.P_rthr").as_double();
    m_twist_cb_pc.m_period_ema.m_alpha = this->get_parameter("pose.ema_alpha").as_double();
    init_ekf();

    // Initialize ROS attributes
    m_pub_pose = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/pose_estimator/" + m_pose_frame + "_filtered", 0);
    m_pub_pose_pred = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/pose_estimator/" + m_pose_frame + "_prediction", 0);
    m_sub_cam_twist = this->create_subscription<geometry_msgs::msg::TwistStamped>(
        "/camera_twist", 0, std::bind(&PoseEstimator::callback_cam_twist, this, _1));
    m_sub_pose = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/pose", 0, std::bind(&PoseEstimator::callback_pose, this, _1));
    m_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    m_cbh_param =
        this->add_on_set_parameters_callback(std::bind(&PoseEstimator::callback_params, this, _1));
}

void PoseEstimator::publish_pose(const std_msgs::msg::Header &header, const bool pred)
{
    State x;
    Covariance P;
    m_ekf.getStateAndCovariance(x, P);
    if (P.topLeftCorner<3, 3>().maxCoeff() > m_pose_P_thr.first ||
        P.bottomRightCorner<3, 3>().maxCoeff() > m_pose_P_thr.second)
    {
        return;
    }

    geometry_msgs::msg::PoseStamped msg;
    msg.header = header;
    msg.pose = tf2::toMsg(x.isometry());
    if (pred)
        m_pub_pose_pred->publish(msg);
    else
        m_pub_pose->publish(msg);
}

void PoseEstimator::make_pose_tf(const std_msgs::msg::Header &header, const bool pred)
{
    State x;
    Covariance P;
    m_ekf.getStateAndCovariance(x, P);
    if (P.topLeftCorner<3, 3>().maxCoeff() > m_pose_P_thr.first ||
        P.bottomRightCorner<3, 3>().maxCoeff() > m_pose_P_thr.second)
    {
        return;
    }

    geometry_msgs::msg::TransformStamped transform = tf2::eigenToTransform(x.isometry());
    transform.header = header;
    transform.child_frame_id = m_pose_frame + (pred ? "p" : "f");
    m_tf_broadcaster->sendTransform(transform);
}

void PoseEstimator::callback_cam_twist(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
    if (!m_ekf_init) return;
    // Update callback period EMA
    rclcpp::Time stamp_now = this->get_clock()->now();
    m_twist_cb_pc.update(stamp_now);

    // Update pose through EKF prediction step
    std::optional<double> dt = m_twist_cb_pc.get();
    if (!dt.has_value()) return;
    Action cam_twist;
    tf2::fromMsg(msg->twist, cam_twist);
    m_ekf.predict(cam_twist * (*dt), action_fn);

    // Publish pose and transform
    publish_pose(msg->header, true);
    make_pose_tf(msg->header, true);
}

void PoseEstimator::callback_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    // Update pose through EKF update step
    Measurement y = utils::geometry::to_mnf_se3<double, true>(msg->pose);
    if (m_ekf_init)
    {
        m_ekf.update(y);
    }
    else
    {
        m_ekf_init = true;
        m_ekf.setState(y);
    }

    // Publish pose and transform
    publish_pose(msg->header, false);
    make_pose_tf(msg->header, false);
}

rcl_interfaces::msg::SetParametersResult
PoseEstimator::callback_params(const std::vector<rclcpp::Parameter> &parameters)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";
    auto check_cov = [&result](const rclcpp::Parameter &p)
    {
        if (p.get_type() != rclcpp::PARAMETER_DOUBLE_ARRAY ||
            p.as_double_array().size() != Covariance::RowsAtCompileTime)
        {
            result.successful = false;
            result.reason = p.get_name() + " must be a double array of size " +
                            std::to_string(Covariance::RowsAtCompileTime);
            return false;
        }
        return true;
    };

    // Check for params that are allowed to change at runtime
    for (const auto &param : parameters)
    {
        // EKF covariance diagonal parameters
        if (param.get_name() == "ekf.P0_diag" || param.get_name() == "ekf.Q_diag" ||
            param.get_name() == "ekf.R_diag")
        {
            if (!check_cov(param)) break;
        }
        else
        {
            result.successful = false;
            result.reason = param.get_name() + " not allowed to change at runtime.";
            break;
        }
    }
    if (result.successful)
    {
        init_ekf(); // re-intialize EKF
    }
    return result;
}

void PoseEstimator::init_ekf()
{
    std::vector<double> P0_diag = this->get_parameter("ekf.P0_diag").as_double_array();
    std::vector<double> Q_diag = this->get_parameter("ekf.Q_diag").as_double_array();
    std::vector<double> R_diag = this->get_parameter("ekf.R_diag").as_double_array();

    Covariance ekf_P0, ekf_Q, ekf_R;
    ekf_P0.setZero();
    ekf_Q.setZero();
    ekf_R.setZero();

    ekf_P0.diagonal() = Eigen::Map<Eigen::VectorXd>(P0_diag.data(), P0_diag.size());
    ekf_Q.diagonal() = Eigen::Map<Eigen::VectorXd>(Q_diag.data(), Q_diag.size());
    ekf_R.diagonal() = Eigen::Map<Eigen::VectorXd>(R_diag.data(), R_diag.size());

    m_ekf_init = false;
    m_ekf.setErrorCovariance(ekf_P0);
    m_ekf.setProcessCovariance(ekf_Q);
    m_ekf.setMeasurementCovariance(ekf_R);
}

RCLCPP_COMPONENTS_REGISTER_NODE(PoseEstimator)
