#include "ibvs_controller.hpp"

IbvsController::IbvsController() : Node("ibvs_controller")
{
    // Declare ROS parameters
    this->declare_parameter("verbose", rclcpp::PARAMETER_BOOL);
    this->declare_parameter("robot_description", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.base_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.ee_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("frame.cam_frame", rclcpp::PARAMETER_STRING);
    this->declare_parameter("tag.tag_family", rclcpp::PARAMETER_STRING);
    this->declare_parameter("tag.tag_size", rclcpp::PARAMETER_DOUBLE);
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
    this->declare_parameter("ctrl.lambda", rclcpp::PARAMETER_DOUBLE_ARRAY);

    // Initialize non-ROS class attributes
    m_is_cam_init = false;
    m_base_frame = this->get_parameter("frame.base_frame").as_string();
    m_ee_frame = this->get_parameter("frame.ee_frame").as_string();
    m_cam_frame = this->get_parameter("frame.cam_frame").as_string();
    m_tag_family = this->get_parameter("tag.tag_family").as_string();
    double tag_size = this->get_parameter("tag.tag_size").as_double();
    std::vector<int64_t> tag_ids = this->get_parameter("tag.tag_ids").as_integer_array();
    std::transform(tag_ids.begin(), tag_ids.end(), std::back_inserter(m_tag_ids),
                   [](int64_t id) { return static_cast<int>(id); });
    m_points[0].setWorldCoordinates(-tag_size / 2, -tag_size / 2, 0);
    m_points[1].setWorldCoordinates(tag_size / 2, -tag_size / 2, 0);
    m_points[2].setWorldCoordinates(tag_size / 2, tag_size / 2, 0);
    m_points[3].setWorldCoordinates(-tag_size / 2, tag_size / 2, 0);
    for (const int id : m_tag_ids)
    {
        m_p.insert({id, std::array<vpFeaturePoint, 4>()});
    }
    init_robot_and_controller();

    // Initialize ROS attributes
    m_pub_perr =
        this->create_publisher<geometry_msgs::msg::PoseArray>("/ibvs_controller/pose_error", 10);
    m_pub_traj = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
        "/joint_trajectory_controller/joint_trajectory", 0);
    m_sub_js = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 0, std::bind(&IbvsController::callback_js, this, _1));
    m_sub_cam_info = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", 0, std::bind(&IbvsController::callback_cam_info, this, _1));
    m_sub_tag = this->create_subscription<AprilTagDetectionArray>(
        "/detections", 0, std::bind(&IbvsController::callback_tag, this, _1));
    m_tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    m_tf_listener = std::make_shared<tf2_ros::TransformListener>(*m_tf_buffer);
}

IbvsController::~IbvsController()
{
    std::vector<double> qdot_zero(m_robot.getNumDofs(), 0.0);
    publish_traj(qdot_zero);
}

void IbvsController::post_init()
{
    constexpr int max_trials = 20;
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

void IbvsController::publish_perr(const std::vector<double> &perr)
{
    geometry_msgs::msg::PoseArray msg;
    msg.header.stamp = this->get_clock()->now();

    std::size_t n_poses = perr.size() / 2;
    msg.poses.resize(n_poses);
    for (std::size_t i = 0; i < n_poses; i++)
    {
        msg.poses[i].position.x = perr[2 * i];
        msg.poses[i].position.y = perr[2 * i + 1];
    }
    m_pub_perr->publish(msg);
}

void IbvsController::publish_traj(const std::vector<double> &qdot)
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

void IbvsController::callback_js(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    m_joint_names = msg->name;
    m_robot.setJointPosition(msg->position);
}

void IbvsController::callback_cam_info(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    if (!m_is_cam_init)
    {
        // K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        m_cam_params.initPersProjWithoutDistortion(msg->k[0], msg->k[4], msg->k[2], msg->k[5]);
        m_is_cam_init = true;
    }
}

void IbvsController::callback_tag(const AprilTagDetectionArray::SharedPtr msg)
{
    if (m_is_cam_init && m_pd.size() > 0)
    {
        // Update tag feature points
        std::vector<int> valid_ids, invalid_ids;
        for (auto const &tag : m_pd)
        {
            const int id = tag.first;
            auto it = std::find_if(msg->detections.cbegin(), msg->detections.cend(),
                                   [id](const AprilTagDetection &dtn) { return dtn.id == id; });
            if (it != msg->detections.cend())
            {
                valid_ids.push_back(id);
                vpHomogeneousMatrix cMo = gm_pose_to_vp_hmatrix((*it).pose.pose.pose);
                for (std::size_t i = 0; i < 4; i++)
                {
                    vpImagePoint corner((*it).corners[i].y, (*it).corners[i].x);
                    vpFeatureBuilder::create(m_p[id][i], m_cam_params, corner);
                    vpColVector cP;
                    m_points[i].changeFrame(cMo, cP);
                    m_p[id][i].set_Z(cP[2]);
                }
            }
            else
            {
                invalid_ids.push_back(id);
            }
        }

        // Replace invalid tags with valid duplicates if available
        if (valid_ids.size() == 0)
        {
            return;
        }
        if (invalid_ids.size() > 0)
        {
            for (const int id : invalid_ids)
            {
                m_p[id] = m_p[valid_ids[0]];
                m_pd[id] = m_pd[valid_ids[0]];
            }
        }

        // Get end-effector pose relative to robot base from TF Tree
        vpHomogeneousMatrix fMe;
        if (!lookup_transform(m_base_frame, m_ee_frame, fMe))
        {
            return;
        }

        // Compute camera velocity and convert to joint velocities
        vpColVector v_c = m_lambda.hadamard(m_controller.computeControlLaw());
        vpColVector perr = m_controller.getError();
        std::vector<double> qdot(m_robot.getNumDofs(), 0.0);
        if (perr.sumSquare() > m_conv_eps)
        {
            qdot = m_robot.computeJointVelocity(fMe, v_c);
        }
        publish_traj(qdot);
        publish_perr(perr.toStdVector()); // log errors
    }

    // Check for desired pose updates
    get_desired_poses();
}

void IbvsController::get_desired_poses()
{
    // Query transforms for tag transforms relative to desired camera frame
    // and update the corresponding desired feature points
    for (const int id : m_tag_ids)
    {
        std::string tag_id_frame = m_tag_family + ":" + std::to_string(id);
        std::string cam_d_frame = m_cam_frame + ":" + std::to_string(id);
        vpHomogeneousMatrix cdMo_id;
        if (lookup_transform(cam_d_frame, tag_id_frame, cdMo_id))
        {
            // Initialize new tag for controller to track
            if (m_pd.find(id) == m_pd.end())
            {
                m_pd.insert({id, std::array<vpFeaturePoint, 4>()});
                for (std::size_t i = 0; i < 4; i++)
                {
                    m_controller.addFeature(m_p[id][i], m_pd[id][i]);
                }
            }

            // Update tag desired feature points
            for (std::size_t i = 0; i < 4; i++)
            {
                m_points[i].track(cdMo_id);
                vpFeatureBuilder::create(m_pd[id][i], m_points[i]);
            }
        }
    }
}

bool IbvsController::lookup_transform(const std::string &target_frame,
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

void IbvsController::init_robot_and_controller()
{
    bool verbose = this->get_parameter("verbose").as_bool();
    std::string robot_description = this->get_parameter("robot_description").as_string();

    // Initialize IK solver
    double ik_eps = this->get_parameter("ik.eps").as_double();
    double ik_lambda = this->get_parameter("ik.lambda").as_double();
    int ik_max_iters = static_cast<int>(this->get_parameter("ik.max_iters").as_int());
    std::vector<double> ik_weight_js_vec = this->get_parameter("ik.weight_js").as_double_array();

    int js_size = static_cast<int>(std::sqrt(static_cast<double>(ik_weight_js_vec.size())));
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(
        ik_weight_js_vec.data(), js_size, js_size);
    Eigen::MatrixXd ik_weight_js = map;
    vc::solver::KdlIkSolverVel_wlds solver(verbose, m_base_frame, m_ee_frame, ik_eps, ik_lambda,
                                           ik_max_iters, ik_weight_js);

    // Initialize robot
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

    // Initialize controller
    m_conv_eps = std::pow(this->get_parameter("ctrl.conv_ttol").as_double(), 2);
    m_lambda = this->get_parameter("ctrl.lambda").as_double_array();
    m_controller.setLambda(1.0);
    m_controller.setServo(vpServo::EYEINHAND_CAMERA);
    m_controller.setInteractionMatrixType(vpServo::CURRENT);
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto ibvs_controller = std::make_shared<IbvsController>();
    ibvs_controller->post_init();
    rclcpp::spin(ibvs_controller);
    rclcpp::shutdown();
    return 0;
}