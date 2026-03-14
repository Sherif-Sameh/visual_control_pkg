#include "control_pkg/pose_controller.hpp"

PoseController::PoseController() : Node("pose_controller")
{
    // Declare ROS parameters
    this->declare_parameter("verbose", rclcpp::PARAMETER_BOOL);
    this->declare_parameter("robot_description", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.base_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.ee_frame", rclcpp::PARAMETER_STRING);
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
    init_robot();
    init_controller();

    // Initialize ROS attributes
    m_pub_traj = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
        "/joint_trajectory_controller/joint_trajectory", 0);
    m_pub_perr =
        this->create_publisher<geometry_msgs::msg::PoseArray>("/pose_controller/pose_error", 10);
    m_sub_gp = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 0, std::bind(&PoseController::callback_gp, this, _1));
    m_sub_js = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 0, std::bind(&PoseController::callback_js, this, _1));
    m_tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    m_tf_listener = std::make_shared<tf2_ros::TransformListener>(*m_tf_buffer);
    m_cbh_param =
        this->add_on_set_parameters_callback(std::bind(&PoseController::callback_params, this, _1));
}

PoseController::~PoseController()
{
    std::vector<double> qdot_zero(m_robot.getNumDofs(), 0.0);
    publish_traj(qdot_zero);
}

void PoseController::publish_traj(const std::vector<double> &qdot)
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

void PoseController::publish_perr(const vpPoseVector &edPe)
{
    geometry_msgs::msg::PoseArray msg;
    msg.header.stamp = this->get_clock()->now();

    msg.poses.resize(1);
    msg.poses[0].position.x = edPe[0];
    msg.poses[0].position.y = edPe[1];
    msg.poses[0].position.z = edPe[2];
    msg.poses[0].orientation = utils::geometry::xyz_aa_to_gm_quat(edPe[3], edPe[4], edPe[5]);
    m_pub_perr->publish(msg);
}

void PoseController::callback_gp(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    m_fMed = utils::geometry::gm_pose_to_vp_hmatrix(msg->pose);
}

void PoseController::callback_js(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (!m_fMed.has_value())
    {
        RCLCPP_DEBUG_STREAM(this->get_logger(), "No desired pose available.");
        return;
    }

    m_joint_names = msg->name;
    m_robot.setJointPosition(msg->position);
    vpHomogeneousMatrix fMe;
    if (!utils::ros_tf2::lookup_transform(m_base_frame, m_ee_frame, m_tf_buffer, fMe)) return;

    vpPoseVector edPe;
    edPe.buildFrom(m_fMed.value().inverse() * fMe);
    vpColVector v_e(6, 0.0);
    if (!has_converged(edPe))
    {
        v_e = -m_lambda.hadamard(edPe);
    }
    std::vector<double> qdot = m_robot.computeJointVelocity(fMe, v_e);
    publish_traj(qdot);
    publish_perr(edPe);
}

rcl_interfaces::msg::SetParametersResult
PoseController::callback_params(const std::vector<rclcpp::Parameter> &parameters)
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
            m_conv_eps.first = std::pow(param.as_double(), 2);
        }
        if (param.get_name() == "ctrl.conv_rtol")
        {
            if (param.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE)
            {
                failed("conv_rtol must be double");
                break;
            }
            m_conv_eps.second = std::pow(param.as_double(), 2);
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

void PoseController::init_robot()
{
    // Initialize IK solver
    bool verbose = this->get_parameter("verbose").as_bool();
    double ik_eps = this->get_parameter("ik.eps").as_double();
    double ik_lambda = this->get_parameter("ik.lambda").as_double();
    int ik_max_iters = static_cast<int>(this->get_parameter("ik.max_iters").as_int());
    std::vector<double> vec = this->get_parameter("ik.weight_js").as_double_array();
    Eigen::MatrixXd ik_weight_js;
    utils::mappings::vec_to_sqr_eigen_matrix<double, Eigen::StorageOptions::RowMajor>(vec,
                                                                                      ik_weight_js);
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
    m_robot.set_eMc(vpHomogeneousMatrix()); // set to identity (no camera)
}

void PoseController::init_controller()
{
    m_conv_eps.first = std::pow(this->get_parameter("ctrl.conv_ttol").as_double(), 2);
    m_conv_eps.second = std::pow(this->get_parameter("ctrl.conv_rtol").as_double(), 2);
    m_lambda = this->get_parameter("ctrl.lambda").as_double_array();
}

bool PoseController::has_converged(const vpPoseVector &edPe)
{
    return (edPe.getTranslationVector().sumSquare() < m_conv_eps.first &&
            edPe.getThetaUVector().sumSquare() < m_conv_eps.second);
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto pose_controller = std::make_shared<PoseController>();
    rclcpp::spin(pose_controller);
    rclcpp::shutdown();
    return 0;
}