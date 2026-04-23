#include "control_pkg/pbvs_controller.hpp"

PbvsController::PbvsController()
    : Node("pbvs_controller"),
      m_pf{vpFeatureTranslation(vpFeatureTranslation::cdMc), vpFeatureThetaU(vpFeatureThetaU::cdRc)}
{
    // Declare ROS parameters
    this->declare_parameter("verbose", rclcpp::PARAMETER_BOOL);
    this->declare_parameter("timeout", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("robot_description", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.base_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.ee_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.cam_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("tag.tag_id", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("ik.eps", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ik.lambda", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ik.max_iters", rclcpp::PARAMETER_INTEGER);
    this->declare_parameter("ik.weight_js", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("robot.max_tvel", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("robot.max_rvel", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("robot.max_vel_sf", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("robot.max_qdot", rclcpp::PARAMETER_DOUBLE_ARRAY);
    this->declare_parameter("ctrl.lpf_coeff", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ctrl.conv_ttol", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ctrl.conv_rtol", rclcpp::PARAMETER_DOUBLE);
    this->declare_parameter("ctrl.lambda", rclcpp::PARAMETER_DOUBLE_ARRAY);

    // Initialize non-ROS class attributes
    m_timeout = this->get_parameter("timeout").as_double();
    m_base_frame = this->get_parameter("frame.base_frame").as_string();
    m_ee_frame = this->get_parameter("frame.ee_frame").as_string();
    m_cam_frame = this->get_parameter("frame.cam_frame").as_string();
    m_tag_id = static_cast<int>(this->get_parameter("tag.tag_id").as_int());
    m_ctrl_ts = this->get_clock()->now();
    m_pf.m_t.buildFrom(vpHomogeneousMatrix());  // set error to zero
    m_pf.m_tu.buildFrom(vpHomogeneousMatrix()); // set error to zero
    init_robot();
    init_controller();

    // Initialize ROS attributes
    m_timer = this->create_wall_timer(0.2s, std::bind(&PbvsController::callback_timer, this));
    m_pub_traj = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
        "/joint_trajectory_controller/joint_trajectory", 0);
    m_pub_perr =
        this->create_publisher<geometry_msgs::msg::PoseArray>("/pbvs_controller/pose_error", 10);
    m_pub_cam_twist = this->create_publisher<geometry_msgs::msg::TwistStamped>(
        "/pbvs_controller/camera_twist", 0);
    m_sub_js = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 0, std::bind(&PbvsController::callback_js, this, _1));
    m_sub_tag = this->create_subscription<AprilTagDetectionArray>(
        "/detections", 0, std::bind(&PbvsController::callback_tag, this, _1));
    m_sub_traj_des = this->create_subscription<MultiDOFJointTrajectory>(
        "/desired_trajectory", 0, std::bind(&PbvsController::callback_traj_des, this, _1));
    m_tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    m_tf_listener = std::make_shared<tf2_ros::TransformListener>(*m_tf_buffer);
    m_cbh_param =
        this->add_on_set_parameters_callback(std::bind(&PbvsController::callback_params, this, _1));
}

PbvsController::~PbvsController()
{
    std::vector<double> qdot_zero(m_robot.getNumDofs(), 0.0);
    publish_traj(qdot_zero);
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
            utils::geometry::to_gm_quat(perr[6 * i + 3], perr[6 * i + 4], perr[6 * i + 5]);
    }
    m_pub_perr->publish(msg);
}

void PbvsController::publish_cam_twist(const vpColVector &v_c)
{
    geometry_msgs::msg::TwistStamped msg;
    msg.header.frame_id = m_cam_frame;
    msg.header.stamp = this->get_clock()->now();

    msg.twist.linear.x = v_c[0];
    msg.twist.linear.y = v_c[1];
    msg.twist.linear.z = v_c[2];
    msg.twist.angular.x = v_c[3];
    msg.twist.angular.y = v_c[4];
    msg.twist.angular.z = v_c[5];
    m_pub_cam_twist->publish(msg);
}

void PbvsController::callback_timer()
{
    // Finish robot initializaton
    if (!m_robot.isInitialized())
    {
        vpHomogeneousMatrix eMc;
        if (utils::ros_tf2::lookup_transform(m_ee_frame, m_cam_frame, m_tf_buffer, eMc))
        {
            m_robot.set_eMc(eMc);
        }
    }
    // Check for controller timeout
    if ((this->get_clock()->now() - m_ctrl_ts).seconds() > m_timeout)
    {
        std::vector<double> qdot_zero(m_robot.getNumDofs(), 0.0);
        publish_traj(qdot_zero);
    }
}

void PbvsController::callback_js(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    m_joint_names = msg->name;
    m_robot.setJointPosition(msg->position);
}

void PbvsController::callback_tag(const AprilTagDetectionArray::SharedPtr msg)
{
    if (!m_traj_msg) return; // no desired features

    bool valid_id = update_features(msg->detections);
    if (!valid_id) return; // no valid tag

    // Get end-effector pose relative to robot base from TF Tree
    vpHomogeneousMatrix fMe;
    if (!(m_robot.isInitialized() &&
          utils::ros_tf2::lookup_transform(m_base_frame, m_ee_frame, m_tf_buffer, fMe)))
        return;

    // Compute camera velocity and convert to joint velocities
    vpColVector v_c = m_v_cam_ff.m_current;
    if (!has_converged())
    {
        v_c += m_lambda.hadamard(m_controller.computeControlLaw());
    }
    std::vector<double> qdot = m_robot.computeJointVelocity(fMe, v_c);
    publish_traj(qdot);
    publish_perr(m_controller.getError().toStdVector());
    publish_cam_twist(v_c);
    m_ctrl_ts = this->get_clock()->now();
}

void PbvsController::callback_traj_des(const MultiDOFJointTrajectory::SharedPtr msg)
{
    // store for interpolation during detection callbacks
    m_traj_ts = this->get_clock()->now();
    m_traj_msg = msg;
}

rcl_interfaces::msg::SetParametersResult
PbvsController::callback_params(const std::vector<rclcpp::Parameter> &parameters)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";
    bool has_ik_params = false;

    auto failed = [&has_ik_params, &result](const std::string &reason)
    {
        has_ik_params = false;
        result.successful = false;
        result.reason = reason;
    };

    // Check for params that are allowed to change at runtime
    for (const auto &param : parameters)
    {
        // IK solver param
        if (!has_ik_params && param.get_name().find("ik.") != std::string::npos)
        {
            has_ik_params = true;
        }
        // Controller convergence tolerances
        if (param.get_name() == "ctrl.conv_ttol")
        {
            if (param.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE)
            {
                failed("conv_ttol must be double");
                break;
            }
            m_conv_eps.m_ttol = std::pow(param.as_double(), 2);
        }
        if (param.get_name() == "ctrl.conv_rtol")
        {
            if (param.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE)
            {
                failed("conv_rtol must be double");
                break;
            }
            m_conv_eps.m_rtol = std::pow(param.as_double(), 2);
        }
        // Controller lambda (gain)
        if (param.get_name() == "ctrl.lambda")
        {
            if (param.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY ||
                param.as_double_array().size() != m_lambda.size())
            {
                failed("lambda must be a double array of size " + std::to_string(m_lambda.size()));
                break;
            }
            m_lambda = param.as_double_array();
        }
    }
    if (has_ik_params)
    {
        // Reintialize robot to update IK solver's parameters
        init_robot();
    }
    return result;
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
    utils::mappings::to_eigen_matrix<double, Eigen::StorageOptions::RowMajor>(vec, ik_weight_js);
    vc::solver::IkSolverVel_wlds solver(verbose, m_base_frame, m_ee_frame, ik_eps, ik_lambda,
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
    m_conv_eps.m_ttol = std::pow(this->get_parameter("ctrl.conv_ttol").as_double(), 2);
    m_conv_eps.m_rtol = std::pow(this->get_parameter("ctrl.conv_rtol").as_double(), 2);
    m_lambda = this->get_parameter("ctrl.lambda").as_double_array();
    m_v_cam_ff.m_alpha = this->get_parameter("ctrl.lpf_coeff").as_double();
    m_v_cam_ff.m_current.resize(6);
    m_v_cam_ff.m_current = 0.0;
    m_controller.setLambda(1.0);
    m_controller.setServo(vpServo::EYEINHAND_CAMERA);
    m_controller.setInteractionMatrixType(vpServo::CURRENT);
    m_controller.addFeature(m_pf.m_t);
    m_controller.addFeature(m_pf.m_tu);
}

bool PbvsController::update_features(const std::vector<AprilTagDetection> &detections)
{
    // Check detections for tag id
    auto it = std::find_if(detections.cbegin(), detections.cend(),
                           [this](const AprilTagDetection &dtn) { return dtn.id == m_tag_id; });
    if (it == detections.cend()) return false;

    // Interpolate trajectory to get reference
    rclcpp::Duration elapsed = this->get_clock()->now() - m_traj_ts;
    auto pt_it = find_traj_point(elapsed);
    if (pt_it == m_traj_msg->points.cbegin()) return false;
    auto pt_prev_it = std::prev(pt_it);
    manif::SE3d oMcd;
    vpColVector v_cam;
    if (pt_it != m_traj_msg->points.cend())
    {
        double pt_sec = rclcpp::Duration((*pt_it).time_from_start).seconds();
        double pt_prev_sec = rclcpp::Duration((*pt_prev_it).time_from_start).seconds();
        double lambda = (elapsed.seconds() - pt_prev_sec) / (pt_sec - pt_prev_sec);
        // Interpolate pose
        manif::SE3d oMcd_1 = utils::geometry::to_mnf_se3<double, true>((*pt_prev_it).transforms[0]);
        manif::SE3d oMcd_2 = utils::geometry::to_mnf_se3<double, true>((*pt_it).transforms[0]);
        oMcd = oMcd_1.plus(lambda * oMcd_2.minus(oMcd_1));
        // Interpolate twist
        vpColVector v_cam_1 = utils::mappings::to_vp_vpcolvector((*pt_prev_it).velocities[0]);
        vpColVector v_cam_2 = utils::mappings::to_vp_vpcolvector((*pt_it).velocities[0]);
        v_cam = v_cam_1 + lambda * (v_cam_2 - v_cam_1);
    }
    else
    {
        oMcd = utils::geometry::to_mnf_se3<double, true>((*pt_prev_it).transforms[0]);
        v_cam = utils::mappings::to_vp_vpcolvector((*pt_prev_it).velocities[0]);
    }

    // Update features and feed-forward velocity signal
    vpHomogeneousMatrix cdMo = utils::geometry::to_vp_hmatrix(oMcd.inverse());
    vpHomogeneousMatrix cMo = utils::geometry::to_vp_hmatrix((*it).pose.pose.pose);
    vpHomogeneousMatrix cdMc = cdMo * cMo.inverse();
    m_pf.m_t.buildFrom(cdMc);
    m_pf.m_tu.buildFrom(cdMc);
    m_v_cam_ff.update(v_cam);
    return true;
}

std::vector<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>::const_iterator
PbvsController::find_traj_point(const rclcpp::Duration &elapsed)
{
    auto it = std::lower_bound(
        m_traj_msg->points.cbegin(), m_traj_msg->points.cend(), elapsed.nanoseconds(),
        [](const trajectory_msgs::msg::MultiDOFJointTrajectoryPoint &pt, const int64_t &value_ns)
        {
            int64_t pt_ns = rclcpp::Duration(pt.time_from_start).nanoseconds();
            return pt_ns < value_ns;
        });
    return it;
}

bool PbvsController::has_converged()
{
    return m_pf.m_t.get_s().sumSquare() < m_conv_eps.m_ttol &&
           m_pf.m_tu.get_s().sumSquare() < m_conv_eps.m_rtol;
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto pbvs_controller = std::make_shared<PbvsController>();
    rclcpp::spin(pbvs_controller);
    rclcpp::shutdown();
    return 0;
}