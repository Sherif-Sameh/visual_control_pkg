#include "vision_pkg/charuco_detector.hpp"

CharucoDetector::CharucoDetector(const rclcpp::NodeOptions &options)
    : Node("charuco_detector", options)
{
    // Declare ROS parameters
    this->declare_parameter("visualize", rclcpp::PARAMETER_BOOL);
    this->declare_parameter("dict.name", rclcpp::PARAMETER_STRING);
    this->declare_parameter("board.xs", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("board.ys", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("board.sq_len", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("board.mk_len", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("params.at.ws.min", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("params.at.ws.max", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("params.at.ws.step", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("params.at.const", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("params.mpr.min", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("params.mpr.max", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("params.paar", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("params.cr.ws", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("params.cr.mi", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("params.cr.ma", rclcpp::PARAMETER_DOUBLE);

    // Initialize non-ROS class attributes
    m_visualize = this->get_parameter("visualize").as_bool();
    std::string dict_name = this->get_parameter("dict.name").as_string();
    int board_xs = this->get_parameter("board.xs").as_int();
    int board_ys = this->get_parameter("board.ys").as_int();
    double board_sq_len = this->get_parameter("board.sq_len").as_double();
    double board_mk_len = this->get_parameter("board.mk_len").as_double();
    m_board_size << board_xs * board_sq_len, board_ys * board_sq_len;
    m_dict = cv::aruco::getPredefinedDictionary(
        utils::str2enum::cvArucoPredefinedDictionaryNameMap.at(dict_name));
    m_board =
        cv::aruco::CharucoBoard::create(board_xs, board_ys, board_sq_len, board_mk_len, m_dict);
    init_detector_params();

    // Initialize ROS attributes
    m_pub_dtn = this->create_publisher<AprilTagDetectionArray>("/charuco_detector/detections", 0);
    m_pub_image = this->create_publisher<sensor_msgs::msg::Image>("/charuco_detector/image", 0);
    m_sub_cam_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", 0, std::bind(&CharucoDetector::callback_cam_info, this, _1));
    m_sub_image = this->create_subscription<sensor_msgs::msg::Image>(
        "/image", 0, std::bind(&CharucoDetector::callback_image, this, _1));
    m_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    m_cbh_param = this->add_on_set_parameters_callback(
        std::bind(&CharucoDetector::callback_params, this, _1));
}

void CharucoDetector::publish_dtn(const std_msgs::msg::Header &header,
                                  const Eigen::Isometry3d &pose)
{
    AprilTagDetectionArray msg;
    msg.header = header;

    AprilTagDetection board;
    board.family = "charuco";
    board.id = 0;
    board.pose.pose.pose = tf2::toMsg(pose);
    msg.detections.push_back(board);
    m_pub_dtn->publish(msg);
}

void CharucoDetector::publish_dtn_tf(const std_msgs::msg::Header &header,
                                     const Eigen::Isometry3d &pose)
{
    geometry_msgs::msg::TransformStamped transform = tf2::eigenToTransform(pose);
    transform.header = header;
    transform.child_frame_id = "charuco:0";
    m_tf_broadcaster->sendTransform(transform);
}

void CharucoDetector::callback_cam_info(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    if (!m_cam_K.has_value())
    {
        m_cam_K.emplace(3, 3, CV_64F);
    }
    (*m_cam_K) = (cv::Mat_<double>(3, 3) << msg->k[0], msg->k[1], msg->k[2], msg->k[3], msg->k[4],
                  msg->k[5], msg->k[6], msg->k[7], msg->k[8]);
}

void CharucoDetector::callback_image(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (!m_cam_K.has_value()) return;
    // Convert to cv::Mat without copying
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(msg);

    // Detect markers
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2d>> marker_corners;
    cv::aruco::detectMarkers(cv_ptr->image, m_dict, marker_corners, marker_ids, m_params);
    if (marker_ids.size() == 0) return;

    // Detect ChArUco corners
    std::vector<int> charuco_ids;
    std::vector<cv::Point2d> charuco_corners;
    cv::aruco::interpolateCornersCharuco(marker_corners, marker_ids, cv_ptr->image, m_board,
                                         charuco_corners, charuco_ids, *m_cam_K);
    if (charuco_ids.size() == 0) return;

    // Estimate ChArUco pose
    cv::Vec3d rvec, tvec;
    bool valid = cv::aruco::estimatePoseCharucoBoard(charuco_corners, charuco_ids, m_board,
                                                     *m_cam_K, cv::noArray(), rvec, tvec);
    if (!valid) return;

    // Transform to AprilTag convention and publish everything
    auto pose = transform_board_pose(tvec, rvec);
    publish_dtn(msg->header, pose);
    publish_dtn_tf(msg->header, pose);
    if (m_visualize)
    {
        cv::Mat img_cp;
        cv_ptr->image.copyTo(img_cp);
        cv::aruco::drawDetectedMarkers(img_cp, marker_corners, marker_ids);
        cv::aruco::drawDetectedCornersCharuco(img_cp, charuco_corners, charuco_ids);
        cv::drawFrameAxes(img_cp, *m_cam_K, cv::noArray(), rvec, tvec, 0.1f);
        auto msg_out = cv_bridge::CvImage(msg->header, msg->encoding, img_cp).toImageMsg();
        m_pub_image->publish(*msg_out);
    }
}

rcl_interfaces::msg::SetParametersResult
CharucoDetector::callback_params(const std::vector<rclcpp::Parameter> &parameters)
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
        // Check that it's one of the detector's parameters
        if (param.get_name().find("params") == std::string::npos)
        {
            failed(param.get_name() + " cannot be changed at runtime.");
            break;
        }
        // Check all parameters whose type is integer
        if ((param.get_name() == "params.at.ws.min" || param.get_name() == "params.at.ws.max" ||
             param.get_name() == "params.at.ws.step" || param.get_name() == "params.cr.ws" ||
             param.get_name() == "params.cr.mi") &&
            param.get_type() != rclcpp::PARAMETER_INTEGER)
        {
            failed(param.get_name() + " must be integer");
            break;
        }
        // Check all parameters whose type is double
        if ((param.get_name() == "params.at.const" || param.get_name() == "params.mpr.min" ||
             param.get_name() == "params.mpr.max" || param.get_name() == "params.paar" ||
             param.get_name() == "params.cr.ma") &&
            param.get_type() != rclcpp::PARAMETER_DOUBLE)
        {
            failed(param.get_name() + " must be double");
            break;
        }
    }
    if (result.successful)
    {
        init_detector_params(); // re-init all detector params
    }
    return result;
}

void CharucoDetector::init_detector_params()
{
    m_params = cv::aruco::DetectorParameters::create();
    m_params->adaptiveThreshWinSizeMin = this->get_parameter("params.at.ws.min").as_int();
    m_params->adaptiveThreshWinSizeMax = this->get_parameter("params.at.ws.max").as_int();
    m_params->adaptiveThreshWinSizeStep = this->get_parameter("params.at.ws.step").as_int();
    m_params->adaptiveThreshConstant = this->get_parameter("params.at.const").as_double();
    m_params->minMarkerPerimeterRate = this->get_parameter("params.mpr.min").as_double();
    m_params->maxMarkerPerimeterRate = this->get_parameter("params.mpr.max").as_double();
    m_params->polygonalApproxAccuracyRate = this->get_parameter("params.paar").as_double();
    m_params->cornerRefinementWinSize = this->get_parameter("params.cr.ws").as_int();
    m_params->cornerRefinementMaxIterations = this->get_parameter("params.cr.mi").as_int();
    m_params->cornerRefinementMinAccuracy = this->get_parameter("params.cr.ma").as_double();
}

Eigen::Isometry3d CharucoDetector::transform_board_pose(const cv::Vec3d &tvec,
                                                        const cv::Vec3d &rvec)
{
    // Create pose from tvec and rvec
    Eigen::Translation3d t(tvec[0], tvec[1], tvec[2]);
    Eigen::AngleAxisd tu = utils::geometry::to_eigen_angle_axis(rvec[0], rvec[1], rvec[2]);
    Eigen::Isometry3d pose = t * tu;

    // Move pose to board center and flip axes to X (right), Y (down), Z (in)
    Eigen::Translation3d t_offset(m_board_size(0) / 2, -m_board_size(1) / 2, 0);
    Eigen::AngleAxisd tu_offset(M_PI, Eigen::Vector3d::UnitX());
    pose = (pose * t_offset).rotate(tu_offset);
    return pose;
}

RCLCPP_COMPONENTS_REGISTER_NODE(CharucoDetector)
