#include "vpRobotRos.hpp"

namespace vc
{
    namespace visp
    {
        vpRobotRos::vpRobotRos() : vpRobot(), m_is_init(false), m_max_vel_sf(1.0), m_max_vel(6, 0.5)
        {
        }

        vpRobotRos::vpRobotRos(const bool verbose, const double max_tvel, const double max_rvel,
                               const vpColVector &max_qdot,
                               const solver::KdlIkSolverVel_wlds &solver)
            : vpRobot(), m_is_init(false), m_max_vel_sf(1.0), m_q_kdl(max_qdot.size()),
              m_qdot(max_qdot.size(), 0.0), m_max_qdot(max_qdot),
              m_max_vel(
                  std::vector<double>{max_tvel, max_tvel, max_tvel, max_rvel, max_rvel, max_rvel}),
              m_solver(solver)
        {
            // Override max velocities set by parent constructor
            verbose_ = verbose;
            maxTranslationVelocity = max_tvel;
            maxRotationVelocity = max_rvel;

            // Perform value checks on parameters
            if (max_tvel <= 0.0)
            {
                throw(vpRobotException(vpRobotException::badValue,
                                       "Maximum linear velocity must be > 0. Got %.2f", max_tvel));
            }
            if (max_rvel <= 0.0)
            {
                throw(vpRobotException(vpRobotException::badValue,
                                       "Maximum rotational velocity must be > 0. Got %.2f",
                                       max_rvel));
            }
            for (std::size_t i = 0; i < max_qdot.size(); i++)
            {
                if (max_qdot[i] <= 0.0)
                {
                    throw(vpRobotException(
                        vpRobotException::badValue,
                        "Maximum joint velocity must be > 0. Got %.2f at joint %zu", max_qdot[i],
                        i));
                }
            }
        }

        vpRobotRos::~vpRobotRos() {}

        void vpRobotRos::init()
        {
            throw(vpRobotException(vpRobotException::notImplementedError,
                                   "Function not implemented. Use init(const std::string "
                                   "&urdf_description) instead."));
        }

        void vpRobotRos::init(const std::string &urdf_description)
        {
            // Initialize IK solver from URDF description
            m_solver.initIkSolver(urdf_description);

            // Update attributes and perform consistency checks
            nDof = m_solver.getNumJoints();
            setRobotState(vpRobot::vpRobotStateType::STATE_VELOCITY_CONTROL);
            if (static_cast<unsigned int>(nDof) != m_qdot.size())
            {
                throw(vpRobotException(
                    vpRobotException::dimensionError,
                    "Number of DOFs does not match size of qdot column vector. Got "
                    "nDof = %d and qdot size = %d",
                    nDof, m_qdot.size()));
            }
        }

        bool vpRobotRos::isInitialized() const { return m_is_init; }

        void vpRobotRos::get_eJe(vpMatrix &_eJe)
        {
            (void)_eJe;
            throw(vpRobotException(
                vpRobotException::notImplementedError,
                "Function not implemented. ROS Robot does not require Jacobians explicitly. They "
                "are computed internally through the IK solver only."));
        }

        void vpRobotRos::get_fJe(vpMatrix &_fJe)
        {
            (void)_fJe;
            throw(vpRobotException(
                vpRobotException::notImplementedError,
                "Function not implemented. ROS Robot does not require Jacobians explicitly. They "
                "are computed internally through the IK solver only."));
        }

        void vpRobotRos::getDisplacement(const vpRobot::vpControlFrameType frame, vpColVector &q)
        {
            (void)frame;
            (void)q;
            throw(vpRobotException(vpRobotException::notImplementedError,
                                   "Function not implemented. ROS Robot does not compute forward "
                                   "kinematics and relies on TF transformations instead."));
        }

        void vpRobotRos::getPosition(const vpRobot::vpControlFrameType frame, vpColVector &q)
        {
            (void)frame;
            (void)q;
            throw(vpRobotException(vpRobotException::notImplementedError,
                                   "Function not implemented. ROS Robot does not compute forward "
                                   "kinematics and relies on TF transformations instead."));
        }

        int vpRobotRos::getNumDofs() const { return m_solver.getNumJoints(); }

        void vpRobotRos::set_eMc(const vpHomogeneousMatrix &eMc)
        {
            // Compute velocity twist matrix from eMc
            m_eVc.buildFrom(eMc);

            // Update initialized status
            if (m_solver.isInitialized())
            {
                m_is_init = true;
            }
        }

        void vpRobotRos::setPosition(const vpRobot::vpControlFrameType frame, const vpColVector &q)
        {
            (void)frame;
            (void)q;
            throw(vpRobotException(vpRobotException::notImplementedError,
                                   "Function not implemented. ROS Robot does not support position "
                                   "control. Only velocity control is allowed."));
        }

        void vpRobotRos::setMaxVelocitySF(const double max_vel_sf)
        {
            assert(max_vel_sf > 0.0 && max_vel_sf <= 1.0);
            if (max_vel_sf <= 0.0 || max_vel_sf > 1.0)
            {
                if (verbose_)
                {
                    std::cerr
                        << "Warning: Scale factor for maximum velocity must be in the interval "
                           "(0, 1]. Got sf ="
                        << max_vel_sf << std::endl;
                }
                return;
            }
            m_max_vel_sf = max_vel_sf;
        }

        void vpRobotRos::setMaxVelocity(const double max_tvel, const double max_rvel)
        {
            assert(max_tvel > 0.0);
            assert(max_rvel > 0.0);
            if (max_tvel <= 0.0 || max_rvel <= 0.0)
            {
                if (verbose_)
                {
                    std::cerr << "Warning: Maximum linear and rotation velocities must be > 0. Got "
                              << max_tvel << " and " << max_rvel << std::endl;
                }
                return;
            }
            m_max_vel[0] = m_max_vel[1] = m_max_vel[2] = max_tvel;
            m_max_vel[3] = m_max_vel[4] = m_max_vel[5] = max_rvel;
        }

        void vpRobotRos::setMaxJointVelocity(const vpColVector &max_qdot)
        {
            for (std::size_t i = 0; i < max_qdot.size(); i++)
            {
                assert(max_qdot[i] > 0.0);
            }
            m_q_kdl.resize(max_qdot.size());
            m_qdot.resize(max_qdot.size());
            m_max_qdot = max_qdot;
        }

        void vpRobotRos::setIkSolver(const solver::KdlIkSolverVel_wlds &solver)
        {
            m_solver = solver;
        }

        void vpRobotRos::setJointPosition(const std::vector<double> &q)
        {
            // Update joint configuration in KDL JntArray format
            assert(q.size() == static_cast<std::size_t>(m_q_kdl.data.size()));
            for (int i = 0; i < nDof; i++)
            {
                m_q_kdl(i) = q[i];
            }
        }

        std::vector<double> vpRobotRos::computeJointVelocity(const vpHomogeneousMatrix &fMe,
                                                             const vpColVector &vel)
        {
            // Update velocity twist matrix for end-effector to body frame from fMe
            m_fVe.buildFrom(fMe);

            // Apply setVelocity function to update desired joint velocities
            setVelocity(vpRobot::vpControlFrameType::CAMERA_FRAME, vel);
            return m_qdot.toStdVector();
        }

        void vpRobotRos::setVelocity(const vpRobot::vpControlFrameType frame,
                                     const vpColVector &vel)
        {
            (void)frame;

            // Initialization and robot state checks
            assert(m_is_init);
            assert(getRobotState() == vpRobot::vpRobotStateType::STATE_VELOCITY_CONTROL);
            assert(frame == vpRobot::vpControlFrameType::CAMERA_FRAME);

            // Convert velocities to base frame and apply twist saturation limits
            vpColVector vel_base = m_fVe * vpRobot::saturateVelocities(
                                               m_eVc * vel, m_max_vel * m_max_vel_sf, verbose_);

            // Convert to joint velocities using IK solver
            KDL::Twist vel_base_kdl;
            for (std::size_t i = 0; i < 3; i++)
            {
                vel_base_kdl.vel.data[i] = vel_base[i];
                vel_base_kdl.rot.data[i] = vel_base[i + 3];
            }
            KDL::JntArray qdot_kdl(nDof);
            m_solver.solveIk(m_q_kdl, vel_base_kdl, qdot_kdl);

            // Update joint velocity and apply velocity saturation
            for (int i = 0; i < nDof; i++)
            {
                m_qdot[i] = qdot_kdl(i);
            }
            m_qdot = vpRobot::saturateVelocities(m_qdot, m_max_qdot, verbose_);
        }
    } // namespace visp
} // namespace vc