/*
 * Description:
 * Extends vpRobot class from ViSP to facilitate robot control through a ROS-based architecture.
 * For more on the ViSP library, refer to: https://github.com/lagadic/visp
 *
 * This implementation makes several assumptions that deviate from the typical design of vpRobot
 * classes available in ViSP. These assumption can be summarized as the following
 *
 *  1) Poses between the end-effector and base frame as well as between the camera and the
 * end-effector are given to the Robot object and not read from it. This is a natural assumption for
 * a ROS-based robot since these transformation will typically come from the ROS TF tree.
 *
 *  2) Only velocity control mode is allowed for controlling the target robot. This is done since
 * this class is intended for use in Visual Servoing applications, where only velocity control is
 * needed.
 *
 *  3) Velocity commands are allowed to be set only in the camera frame. This also follows from the
 * Visual Servoing assumption.
 *
 *  4) After velocity commands are given through the setVelocity member function, they are converted
 * to joint velocities through an IK solver and stored in a command vector that should be retreived
 * through the getJointVelocity member function and sent to the robot's ROS driver. This is done
 * since not all robot drivers support sending twist commands but all support joint velocities.
 */

#ifndef VP_ROBOT_ROS
#define VP_ROBOT_ROS

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpVelocityTwistMatrix.h>
#include <visp3/robot/vpRobot.h>
#include <visp3/robot/vpRobotException.h>

#include "kdlIkSolverVel_wdls.hpp"
#include "utils/conversions/mappings.hpp"

namespace vc
{
    namespace visp
    {
        class vpRobotRos : public vpRobot
        {
        public:
            vpRobotRos();
            vpRobotRos(const bool verbose, const double max_tvel, const double max_rvel,
                       const vpColVector &max_qdot, const solver::KdlIkSolverVel_wlds &solver);
            ~vpRobotRos() override;

            void init() override;
            void init(const std::string &urdf_description);
            bool isInitialized() const;

            void get_eJe(vpMatrix &_eJe) override;
            void get_fJe(vpMatrix &_fJe) override;
            void getDisplacement(const vpRobot::vpControlFrameType frame, vpColVector &q) override;
            void getPosition(const vpRobot::vpControlFrameType frame, vpColVector &q) override;
            int getNumDofs() const;

            void set_eMc(const vpHomogeneousMatrix &eMc);
            void setPosition(const vpRobot::vpControlFrameType frame,
                             const vpColVector &q) override;
            void setMaxVelocitySF(const double max_vel_sf);
            void setMaxVelocity(const double max_tvel, const double max_rvel);
            void setMaxJointVelocity(const vpColVector &max_qdot);
            void setIkSolver(const solver::KdlIkSolverVel_wlds &solver);
            void setJointPosition(const std::vector<double> &q);

            std::vector<double> computeJointVelocity(const vpHomogeneousMatrix &fMe,
                                                     const vpColVector &vel);

        protected:
            void setVelocity(const vpRobot::vpControlFrameType frame,
                             const vpColVector &vel) override;

        protected:
            bool m_is_init;         // Whether the robot has been initialized yet or not
            double m_max_vel_sf;    // Scale factor for maximum twist velocities.
            KDL::JntArray m_q_kdl;  // Latest joint positions stored in a KDL JntArray
            vpColVector m_qdot;     // Latest computed joint velocity command
            vpColVector m_max_qdot; // Maximum absolute joint velocity for each joint
            vpColVector m_max_vel;  // Maximum absolute twist vector for desired motion
            vpVelocityTwistMatrix
                m_fVe; // Twist conversion matrix from end-effector frame to base frame
            vpVelocityTwistMatrix
                m_eVc; // Twist conversion matrix from camera to end-effector frame
            solver::KdlIkSolverVel_wlds m_solver; // Solver for inverse velocity kinematics
        };
    } // namespace visp
} // namespace vc

#endif