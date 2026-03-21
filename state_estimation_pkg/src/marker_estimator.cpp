#include "state_estimation_pkg/marker_estimator.hpp"

MarkerEstimator::MarkerEstimator(const rclcpp::NodeOptions &options)
    : Node("marker_estimator", options)
{
    // Declare ROS parameters
    this->declare_parameter("tag.timeout", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("tag.P_tthr", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("tag.P_rthr", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("tag.ema_alpha", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("tag.size", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ekf.P0_diag", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("ekf.Q_diag", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("ekf.R_diag", rclcpp::PARAMETER_DOUBLE_ARRAY);

    // Initialize non-ROS class attributes
    m_tag_timeout = this->get_parameter("tag.timeout").as_double();
    m_tag_P_thr.first = this->get_parameter("tag.P_tthr").as_double();
    m_tag_P_thr.second = this->get_parameter("tag.P_rthr").as_double();
    m_tag_cb_pc.m_period_ema.m_alpha = this->get_parameter("tag.ema_alpha").as_double();
    double tag_size = this->get_parameter("tag.size").as_double();
    m_tag_pts[0] << -tag_size / 2, -tag_size / 2, 0;
    m_tag_pts[1] << tag_size / 2, -tag_size / 2, 0;
    m_tag_pts[2] << tag_size / 2, tag_size / 2, 0;
    m_tag_pts[3] << -tag_size / 2, tag_size / 2, 0;
    init_ekf();

    // Initialize ROS attributes
    m_pub_tag =
        this->create_publisher<AprilTagDetectionArray>("/marker_estimator/detections_filtered", 0);
    m_sub_cam_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", 0, std::bind(&MarkerEstimator::callback_cam_info, this, _1));
    m_sub_cam_twist = this->create_subscription<geometry_msgs::msg::TwistStamped>(
        "/camera_twist", 0, std::bind(&MarkerEstimator::callback_cam_twist, this, _1));
    m_sub_tag = this->create_subscription<AprilTagDetectionArray>(
        "/detections", 0, std::bind(&MarkerEstimator::callback_tag, this, _1));
    m_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    m_cbh_param = this->add_on_set_parameters_callback(
        std::bind(&MarkerEstimator::callback_params, this, _1));
}

void MarkerEstimator::publish_tag(const std_msgs::msg::Header &header,
                                  const std::string &tag_family)
{
    if (!m_cam_K.has_value()) return;

    AprilTagDetectionArray msg;
    msg.header = header;
    for (const auto &[id, ekf_stamped] : m_ekf_map)
    {
        State x;
        Covariance P;
        ekf_stamped.m_wrapped.getStateAndCovariance(x, P);
        if (P.topLeftCorner<3, 3>().maxCoeff() > m_tag_P_thr.first ||
            P.bottomRightCorner<3, 3>().maxCoeff() > m_tag_P_thr.second)
        {
            continue;
        };
        msg.detections.push_back(create_tag_detection(tag_family, id, x.isometry(), P));
    }
    m_pub_tag->publish(msg);
}

void MarkerEstimator::make_tag_tfs(const std_msgs::msg::Header &header,
                                   const std::string &tag_family)
{
    std::vector<geometry_msgs::msg::TransformStamped> transforms;
    for (const auto &[id, ekf_stamped] : m_ekf_map)
    {
        State x;
        Covariance P;
        ekf_stamped.m_wrapped.getStateAndCovariance(x, P);
        if (P.topLeftCorner<3, 3>().maxCoeff() > m_tag_P_thr.first ||
            P.bottomRightCorner<3, 3>().maxCoeff() > m_tag_P_thr.second)
        {
            continue;
        };
        geometry_msgs::msg::TransformStamped t = tf2::eigenToTransform(x.isometry());
        t.header = header;
        t.child_frame_id = tag_family + ":" + std::to_string(id) + "f";
        transforms.push_back(t);
    }
    m_tf_broadcaster->sendTransform(transforms);
}

void MarkerEstimator::callback_cam_info(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    if (!m_cam_K.has_value())
    {
        m_cam_K.emplace();
    }
    (*m_cam_K) << msg->k[0], msg->k[1], msg->k[2], msg->k[3], msg->k[4], msg->k[5], msg->k[6],
        msg->k[7], msg->k[8];
}

void MarkerEstimator::callback_cam_twist(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
    std::optional<double> dt = m_tag_cb_pc.get();
    if (!dt.has_value()) return;

    Action cam_twist;
    tf2::fromMsg(msg->twist, cam_twist);
    for (auto &[_, ekf_stamped] : m_ekf_map)
    {
        ekf_stamped.m_wrapped.predict(cam_twist * (*dt), action_fn);
    }
}

void MarkerEstimator::callback_tag(const AprilTagDetectionArray::SharedPtr msg)
{
    if (msg->detections.size() == 0) return;

    // Update callback period EMA
    rclcpp::Time stamp_now = this->get_clock()->now();
    m_tag_cb_pc.update(stamp_now);

    // Update existing and new tag IDs
    for (const auto &tag : msg->detections)
    {
        Measurement y = utils::geometry::to_mnf_se3<double, true>(tag.pose.pose.pose);
        auto it = m_ekf_map.find(tag.id);
        if (it != m_ekf_map.end()) // existing tag IDs
        {
            (*it).second.m_stamp = stamp_now;
            (*it).second.m_wrapped.update(y);
        }
        else // new tag IDs
        {
            using EKF = se::EKF<manif::SE3d, se::ActionSE3Features<double>>;
            m_ekf_map.insert({tag.id, {stamp_now, EKF(m_ekf_P0, m_ekf_Q, m_ekf_R)}});
            m_ekf_map[tag.id].m_wrapped.setState(y);
        }
    }

    // Erase old tags that have timed-out
    for (auto it = m_ekf_map.begin(); it != m_ekf_map.end();)
    {
        if ((stamp_now - (*it).second.m_stamp).seconds() > m_tag_timeout)
            it = m_ekf_map.erase(it);
        else
            ++it;
    }

    // Publish detections message and transforms
    publish_tag(msg->header, msg->detections[0].family);
    make_tag_tfs(msg->header, msg->detections[0].family);
}

rcl_interfaces::msg::SetParametersResult
MarkerEstimator::callback_params(const std::vector<rclcpp::Parameter> &parameters)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";
    auto check_cov = [&result](const rclcpp::Parameter &p, Covariance &cov)
    {
        if (p.get_type() != rclcpp::PARAMETER_DOUBLE_ARRAY ||
            p.as_double_array().size() != cov.RowsAtCompileTime)
        {
            result.successful = false;
            result.reason = p.get_name() + " must be a double array of size " +
                            std::to_string(cov.RowsAtCompileTime);
            return false;
        }
        std::vector<double> diag = p.as_double_array();
        cov.diagonal() = Eigen::Map<Eigen::VectorXd>(diag.data(), diag.size());
        return true;
    };

    // Check for params that are allowed to change at runtime
    bool has_valid_ekf_params = false;
    for (const auto &param : parameters)
    {
        // Initial state error covariance diagonal
        if (param.get_name() == "ekf.P0_diag")
        {
            has_valid_ekf_params = check_cov(param, m_ekf_P0);
            if (!has_valid_ekf_params) break;
        }
        // Process noise covariance diagonal
        if (param.get_name() == "ekf.Q_diag")
        {
            has_valid_ekf_params = check_cov(param, m_ekf_Q);
            if (!has_valid_ekf_params) break;
        }
        // Measurement noise covariance diagonal
        if (param.get_name() == "ekf.R_diag")
        {
            has_valid_ekf_params = check_cov(param, m_ekf_R);
            if (!has_valid_ekf_params) break;
        }
    }
    if (has_valid_ekf_params)
    {
        m_ekf_map.clear(); // clear to re-intialize all EKF instances
    }
    return result;
}

void MarkerEstimator::init_ekf()
{
    std::vector<double> P0_diag = this->get_parameter("ekf.P0_diag").as_double_array();
    std::vector<double> Q_diag = this->get_parameter("ekf.Q_diag").as_double_array();
    std::vector<double> R_diag = this->get_parameter("ekf.R_diag").as_double_array();

    m_ekf_P0.setZero();
    m_ekf_Q.setZero();
    m_ekf_R.setZero();

    m_ekf_P0.diagonal() = Eigen::Map<Eigen::VectorXd>(P0_diag.data(), P0_diag.size());
    m_ekf_Q.diagonal() = Eigen::Map<Eigen::VectorXd>(Q_diag.data(), Q_diag.size());
    m_ekf_R.diagonal() = Eigen::Map<Eigen::VectorXd>(R_diag.data(), R_diag.size());
    m_ekf_map.clear();
}

AprilTagDetection MarkerEstimator::create_tag_detection(const std::string &family, const int id,
                                                        const Eigen::Isometry3d &T_tag_cam,
                                                        const Covariance &cov)
{
    auto img_pts = project_points(T_tag_cam);
    Eigen::Vector2d ctr_pt =
        std::accumulate(img_pts.cbegin(), img_pts.cend(), Eigen::Vector2d::Zero().eval()) / 4;

    AprilTagDetection tag_dtn;
    tag_dtn.family = family;
    tag_dtn.id = id;
    tag_dtn.center.x = ctr_pt.x();
    tag_dtn.center.y = ctr_pt.y();
    for (std::size_t i = 0; i < img_pts.size(); i++)
    {
        tag_dtn.corners[i].x = img_pts[i].x();
        tag_dtn.corners[i].y = img_pts[i].y();
    }
    tag_dtn.pose.pose.pose = tf2::toMsg(T_tag_cam);
    tag_dtn.pose.pose.covariance = utils::mappings::to_array<Covariance, Eigen::RowMajor>(cov);
    return tag_dtn;
}

std::array<Eigen::Vector2d, 4> MarkerEstimator::project_points(const Eigen::Isometry3d &T_tag_cam)
{
    std::array<Eigen::Vector2d, 4> img_pts;
    for (std::size_t i = 0; i < m_tag_pts.size(); i++)
    {
        Eigen::Vector3d img_pt_h = (*m_cam_K) * T_tag_cam * m_tag_pts[i];
        img_pts[i] = img_pt_h.head<2>() / img_pt_h.z();
    }
    return img_pts;
}

RCLCPP_COMPONENTS_REGISTER_NODE(MarkerEstimator)
