#include "pbvs_controller.hpp"

PbvsController::PbvsController()
    : Node("pbvs_controller"),
      cdMo_tmp(vpTranslationVector(0.0, 0.0, 0.5), vpQuaternionVector(0.0, 0.0, 0.0, 1.0))
{
    // Declare ROS parameters
    this->declare_parameter("verbose", rclcpp::PARAMETER_BOOL);
    this->declare_parameter("robot_description", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame/base_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame/ee_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame/cam_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("tag/tag_ids", rclcpp::PARAMETER_INTEGER_ARRAY);
    this->declare_parameter("tag/tag_frames", rclcpp::PARAMETER_STRING_ARRAY);
    this->declare_parameter("ik/eps", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ik/lambda", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ik/max_iters", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("ik/weight_js", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("robot/max_tvel", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("robot/max_rvel", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("robot/max_vel_sf", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("robot/max_qdot", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("ctrl/lambda", rclcpp::PARAMETER_DOUBLE);

    // Initialize non-ROS class attributes
    m_base_frame = this->get_parameter("frame/base_frame").as_string();
    m_ee_frame = this->get_parameter("frame/ee_frame").as_string();
    m_cam_frame = this->get_parameter("frame/cam_frame").as_string();
    m_tag_frames = this->get_parameter("tag/tag_frames").as_string_array();
    std::vector<int64_t> tag_ids = this->get_parameter("tag/tag_ids").as_integer_array();
    std::transform(tag_ids.begin(), tag_ids.end(), std::back_inserter(m_tag_ids),
                   [](int64_t id) { return static_cast<int>(id); });
    for (const int id : m_tag_ids)
    {
        m_t.insert({id, vpFeatureTranslation(vpFeatureTranslation::cdMc)});
        m_tu.insert({id, vpFeatureThetaU(vpFeatureThetaU::cdRc)});
    }
    init_robot_and_controller();
    callback_timer(); // TODO: Remove once callback is properly implemented

    // Initialize ROS attributes
    m_pub_traj = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
        "/joint_trajectory_controller/joint_trajectory", 0);
    m_sub_js = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 0, std::bind(&PbvsController::callback_js, this, _1));
    m_sub_tag = this->create_subscription<AprilTagDetectionArray>(
        "/detections", 0, std::bind(&PbvsController::callback_tag, this, _1));
    m_timer = this->create_wall_timer(1s, std::bind(&PbvsController::callback_timer, this));
    m_tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    m_tf_listener = std::make_shared<tf2_ros::TransformListener>(*m_tf_buffer);
}

PbvsController::~PbvsController()
{
    std::vector<double> qdot_zero(m_robot.getNumDofs(), 0.0);
    publish_traj(qdot_zero);
}

void PbvsController::post_init()
{
    constexpr int max_trials = 10;
    constexpr auto sleep_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(0.5s);
    vpHomogeneousMatrix eMc;
    for (int i = 0; i < max_trials; i++)
    {
        if (lookup_transform(m_ee_frame, m_cam_frame, eMc))
        {
            m_robot.set_eMc(eMc);
            return;
        }
        RCLCPP_WARN_STREAM(this->get_logger(), "Retrying in 0.5s");
        rclcpp::sleep_for(sleep_duration);
    }
    rclcpp::shutdown(nullptr, "Robot initialization failed.");
}

void PbvsController::publish_traj(const std::vector<double> &qdot)
{
    trajectory_msgs::msg::JointTrajectory msg;
    msg.header.stamp = this->get_clock()->now();
    msg.joint_names = m_joint_names;

    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions = std::vector<double>(qdot.size(), 0.0);
    pt.velocities = qdot;
    pt.accelerations = std::vector<double>(qdot.size(), 0.0);
    pt.effort = std::vector<double>(qdot.size(), 0.0);
    pt.time_from_start = rclcpp::Duration::from_seconds(0);

    msg.points.push_back(pt);
    m_pub_traj->publish(msg);
}

void PbvsController::callback_js(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    m_joint_names = msg->name;
    m_robot.setJointPosition(msg->position);
}

void PbvsController::callback_tag(const AprilTagDetectionArray::SharedPtr msg)
{
    // Update tag transformations
    for (int const id : m_tag_ids)
    {
        vpHomogeneousMatrix cdMc; // set error to identity
        auto it = std::find_if(msg->detections.cbegin(), msg->detections.cend(),
                               [id](const AprilTagDetection &dtn) { return dtn.id == id; });
        if (it != msg->detections.cend())
        {
            vpHomogeneousMatrix cMo = gm_pose_to_vp_hmatrix((*it).pose.pose.pose);
            cdMc = m_cdMo[id] * cMo.inverse();
        }
        m_t[id].buildFrom(cdMc);
        m_tu[id].buildFrom(cdMc);
    }

    // Get end-effector pose relative to robot base from TF Tree
    vpHomogeneousMatrix fMe;
    if (!lookup_transform(m_base_frame, m_ee_frame, fMe))
    {
        return;
    }

    // Compute camera velocity and convert to joint velocities
    vpColVector v_c = m_controller.computeControlLaw();
    std::vector<double> qdot = m_robot.computeJointVelocity(fMe, v_c);
    publish_traj(qdot);

    // TODO: Add error logging and tolerance for control activation
}

void PbvsController::callback_timer()
{
    // TODO:
    // Query transforms for tag transforms relative to desired camera frame
    // and update the corresponding homogeneous transforms

    // TODO: Remove when dynamic target transforms are implemented
    for (const int id : m_tag_ids)
    {
        m_cdMo.insert({id, cdMo_tmp});
    }
}

bool PbvsController::lookup_transform(const std::string &target_frame,
                                      const std::string &source_frame, vpHomogeneousMatrix &t)
{
    geometry_msgs::msg::TransformStamped t_gm;
    try
    {
        t_gm = m_tf_buffer->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
    }
    catch (const tf2::TransformException &ex)
    {
        RCLCPP_WARN_STREAM(this->get_logger(), "Could not transform " << target_frame << " to "
                                                                      << source_frame << ": "
                                                                      << ex.what());
        return false;
    }
    t = gm_transform_to_vp_hmatrix(t_gm.transform);
    return true;
}

void PbvsController::init_robot_and_controller()
{
    bool verbose = this->get_parameter("verbose").as_bool();
    std::string robot_description = this->get_parameter("robot_description").as_string();

    // Initialize IK solver
    double ik_eps = this->get_parameter("ik/eps").as_double();
    double ik_lambda = this->get_parameter("ik/lambda").as_double();
    int ik_max_iters = static_cast<int>(this->get_parameter("ik/max_iters").as_int());
    std::vector<double> ik_weight_js_vec = this->get_parameter("ik/weight_js").as_double_array();

    int js_size = static_cast<int>(std::sqrt(static_cast<double>(ik_weight_js_vec.size())));
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(
        ik_weight_js_vec.data(), js_size, js_size);
    Eigen::MatrixXd ik_weight_js = map;
    vc::solver::KdlIkSolverVel_wlds solver(verbose, m_base_frame, m_ee_frame, ik_eps, ik_lambda,
                                           ik_max_iters, ik_weight_js);

    // Initialize robot
    double robot_max_tvel = this->get_parameter("robot/max_tvel").as_double();
    double robot_max_rvel = this->get_parameter("robot/max_rvel").as_double();
    double robot_max_vel_sf = this->get_parameter("robot/max_vel_sf").as_double();
    std::vector<double> robot_max_qdot = this->get_parameter("robot/max_qdot").as_double_array();
    m_robot.setVerbose(verbose);
    m_robot.setMaxVelocity(robot_max_tvel, robot_max_rvel);
    m_robot.setMaxVelocitySF(robot_max_vel_sf);
    m_robot.setMaxJointVelocity(vpColVector(robot_max_qdot));
    m_robot.setIkSolver(solver);
    m_robot.init(robot_description);

    // Initialize controller
    double ctrl_lambda = this->get_parameter("ctrl/lambda").as_double();
    m_controller.setLambda(ctrl_lambda);
    m_controller.setServo(vpServo::EYEINHAND_CAMERA);
    m_controller.setInteractionMatrixType(vpServo::CURRENT);
    for (const int id : m_tag_ids)
    {
        m_controller.addFeature(m_t[id]);
        m_controller.addFeature(m_tu[id]);
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto pbvs_controller = std::make_shared<PbvsController>();
    pbvs_controller->post_init();
    rclcpp::spin(pbvs_controller);
    rclcpp::shutdown();
    return 0;
}