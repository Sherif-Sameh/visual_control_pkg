#include "pbvs_controller.hpp"

PbvsController::PbvsController() : Node("pbvs_controller")
{
    // Declare ROS parameters
    this->declare_parameter("verbose", rclcpp::PARAMETER_BOOL);
    this->declare_parameter("robot_description", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.base_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.ee_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.cam_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("tag.tag_family", rclcpp::PARAMETER_STRING);
    this->declare_parameter("tag.tag_ids", rclcpp::PARAMETER_INTEGER_ARRAY);
    this->declare_parameter("ik.eps", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ik.lambda", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ik.max_iters", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("ik.weight_js", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("robot.max_tvel", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("robot.max_rvel", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("robot.max_vel_sf", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("robot.max_qdot", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("ctrl.conv_ttol", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ctrl.conv_rtol", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ctrl.lambda", rclcpp::PARAMETER_DOUBLE_ARRAY);

    // Initialize non-ROS class attributes
    m_base_frame = this->get_parameter("frame.base_frame").as_string();
    m_ee_frame = this->get_parameter("frame.ee_frame").as_string();
    m_cam_frame = this->get_parameter("frame.cam_frame").as_string();
    m_tag_family = this->get_parameter("tag.tag_family").as_string();
    std::vector<int64_t> tag_ids = this->get_parameter("tag.tag_ids").as_integer_array();
    std::transform(tag_ids.begin(), tag_ids.end(), std::back_inserter(m_tag_ids),
                   [](int64_t id) { return static_cast<int>(id); });
    for (const int id : m_tag_ids)
    {
        m_t.insert({id, vpFeatureTranslation(vpFeatureTranslation::cdMc)});
        m_tu.insert({id, vpFeatureThetaU(vpFeatureThetaU::cdRc)});
        m_t[id].buildFrom(vpHomogeneousMatrix());  // set error to zero
        m_tu[id].buildFrom(vpHomogeneousMatrix()); // set error to zero
    }
    init_robot();
    init_controller();

    // Initialize ROS attributes
    m_setup_timer = this->create_wall_timer(0.25s, std::bind(&PbvsController::post_init, this));
    m_pub_perr =
        this->create_publisher<geometry_msgs::msg::PoseArray>("/pbvs_controller/pose_error", 10);
    m_pub_traj = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
        "/joint_trajectory_controller/joint_trajectory", 0);
    m_sub_js = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 0, std::bind(&PbvsController::callback_js, this, _1));
    m_sub_tag = this->create_subscription<AprilTagDetectionArray>(
        "/detections", 0, std::bind(&PbvsController::callback_tag, this, _1));
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
    vpHomogeneousMatrix eMc;
    if (!lookup_transform(m_ee_frame, m_cam_frame, eMc)) return;

    // Finish initialization and cancel timer
    m_robot.set_eMc(eMc);
    m_setup_timer->cancel();
}

void PbvsController::publish_traj(const std::vector<double> &qdot)
{
    trajectory_msgs::msg::JointTrajectory msg;
    msg.header.stamp = this->get_clock()->now();
    msg.joint_names = m_joint_names;

    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.velocities = qdot;
    pt.time_from_start = rclcpp::Duration::from_seconds(0);

    msg.points.push_back(pt);
    m_pub_traj->publish(msg);
}

void PbvsController::publish_perr(const std::vector<double> &perr)
{
    geometry_msgs::msg::PoseArray msg;
    msg.header.stamp = this->get_clock()->now();

    std::size_t n_poses = perr.size() / 6;
    msg.poses.resize(n_poses);
    for (std::size_t i = 0; i < n_poses; i++)
    {
        msg.poses[i].position.x = perr[6 * i];
        msg.poses[i].position.y = perr[6 * i + 1];
        msg.poses[i].position.z = perr[6 * i + 2];

        msg.poses[i].orientation =
            geometry::xyz_aa_to_gm_quat(perr[6 * i + 3], perr[6 * i + 4], perr[6 * i + 5]);
    }
    m_pub_perr->publish(msg);
}

void PbvsController::callback_js(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    m_joint_names = msg->name;
    m_robot.setJointPosition(msg->position);
}

void PbvsController::callback_tag(const AprilTagDetectionArray::SharedPtr msg)
{
    update_desired_features();
    if (m_cdMo.size() == 0)
    {
        RCLCPP_DEBUG_STREAM(this->get_logger(), "No desired features available.");
        return;
    }

    std::vector<int> valid_ids, invalid_ids;
    update_features(msg->detections, valid_ids, invalid_ids);
    if (valid_ids.size() == 0)
    {
        RCLCPP_DEBUG_STREAM(this->get_logger(), "No valid tags detected.");
        return;
    }
    if (invalid_ids.size() > 0)
    {
        // Replace invalid tags with valid duplicates if available
        for (const int id : invalid_ids)
        {
            m_t[id] = m_t[valid_ids[0]];
            m_tu[id] = m_tu[valid_ids[0]];
        }
    }

    // Get end-effector pose relative to robot base from TF Tree
    vpHomogeneousMatrix fMe;
    if (!(m_robot.isInitialized() && lookup_transform(m_base_frame, m_ee_frame, fMe))) return;

    // Compute camera velocity and convert to joint velocities
    vpColVector v_c = m_lambda.hadamard(m_controller.computeControlLaw());
    std::vector<double> qdot = m_robot.computeJointVelocity(fMe, v_c);
    publish_traj(qdot);
    publish_perr(m_controller.getError().toStdVector()); // log errors
}

void PbvsController::init_robot()
{
    // Initialize IK solver
    bool verbose = this->get_parameter("verbose").as_bool();
    double ik_eps = this->get_parameter("ik.eps").as_double();
    double ik_lambda = this->get_parameter("ik.lambda").as_double();
    int ik_max_iters = static_cast<int>(this->get_parameter("ik.max_iters").as_int());
    std::vector<double> vec = this->get_parameter("ik.weight_js").as_double_array();
    Eigen::MatrixXd ik_weight_js;
    mappings::vec_to_sqr_eigen_matrix<double, Eigen::StorageOptions::RowMajor>(vec, ik_weight_js);
    vc::solver::KdlIkSolverVel_wlds solver(verbose, m_base_frame, m_ee_frame, ik_eps, ik_lambda,
                                           ik_max_iters, ik_weight_js);

    // Initialize robot
    std::string robot_description = this->get_parameter("robot_description").as_string();
    double robot_max_tvel = this->get_parameter("robot.max_tvel").as_double();
    double robot_max_rvel = this->get_parameter("robot.max_rvel").as_double();
    double robot_max_vel_sf = this->get_parameter("robot.max_vel_sf").as_double();
    std::vector<double> robot_max_qdot = this->get_parameter("robot.max_qdot").as_double_array();
    m_robot.setVerbose(verbose);
    m_robot.setMaxVelocity(robot_max_tvel, robot_max_rvel);
    m_robot.setMaxVelocitySF(robot_max_vel_sf);
    m_robot.setMaxJointVelocity(vpColVector(robot_max_qdot));
    m_robot.setIkSolver(solver);
    m_robot.init(robot_description);
}

void PbvsController::init_controller()
{
    m_conv_eps.first = std::pow(this->get_parameter("ctrl.conv_ttol").as_double(), 2);
    m_conv_eps.second = std::pow(this->get_parameter("ctrl.conv_rtol").as_double(), 2);
    m_lambda = this->get_parameter("ctrl.lambda").as_double_array();
    m_controller.setLambda(1.0);
    m_controller.setServo(vpServo::EYEINHAND_CAMERA);
    m_controller.setInteractionMatrixType(vpServo::CURRENT);
    for (const int id : m_tag_ids)
    {
        m_controller.addFeature(m_t[id]);
        m_controller.addFeature(m_tu[id]);
    }
}

void PbvsController::update_desired_features()
{
    // Query transforms for tag transforms relative to desired camera frame
    // and update the corresponding homogeneous transforms
    for (const int id : m_tag_ids)
    {
        std::string tag_id_frame = m_tag_family + ":" + std::to_string(id);
        std::string cam_d_frame = m_cam_frame + ":" + std::to_string(id);
        vpHomogeneousMatrix cdMo_id;
        if (lookup_transform(cam_d_frame, tag_id_frame, cdMo_id))
        {
            m_cdMo[id] = cdMo_id;
        }
    }
}

void PbvsController::update_features(const std::vector<AprilTagDetection> &detections,
                                     std::vector<int> &valid_ids, std::vector<int> &invalid_ids)
{
    for (const auto &[id, cdMo] : m_cdMo)
    {
        invalid_ids.push_back(id);
        auto it = std::find_if(detections.cbegin(), detections.cend(),
                               [id](const AprilTagDetection &dtn) { return dtn.id == id; });
        if (it != detections.cend())
        {
            vpHomogeneousMatrix cMo = geometry::gm_pose_to_vp_hmatrix((*it).pose.pose.pose);
            vpHomogeneousMatrix cdMc = cdMo * cMo.inverse();
            if (cdMc.getTranslationVector().sumSquare() > m_conv_eps.first ||
                cdMc.getThetaUVector().sumSquare() > m_conv_eps.second)
            {
                valid_ids.push_back(id);
                invalid_ids.pop_back();
                m_t[id].buildFrom(cdMc);
                m_tu[id].buildFrom(cdMc);
            }
        }
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
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Could not transform " << target_frame << " to "
                                                                       << source_frame << ": "
                                                                       << ex.what());
        return false;
    }
    t = geometry::gm_transform_to_vp_hmatrix(t_gm.transform);
    return true;
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto pbvs_controller = std::make_shared<PbvsController>();
    rclcpp::spin(pbvs_controller);
    rclcpp::shutdown();
    return 0;
}