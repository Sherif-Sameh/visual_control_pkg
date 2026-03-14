/*
 * Description:
 * Wrapper around the ChainIkSolverVel_wdls inverse velocity kinematics solver from OROCOS KDL.
 * For more, check out the OROCOS KDL project: https://github.com/orocos/orocos_kinematics_dynamics
 */

#ifndef ROBOT_IK_SOLVER_VEL_WDLS
#define ROBOT_IK_SOLVER_VEL_WDLS

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <kdl/chain.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>

#include <visp3/core/vpException.h>

namespace vc
{
    namespace solver
    {
        /**
         * @brief Wrapper around `KDL::ChainIkSolverVel_wdls` inverse velocity kinematics solver
         * from OROCOS KDL.
         *
         * To initialize the solver, the `initIkSolver()` member function should be called with a
         * URDF description string. Afterwards, the IK solver can be used normally provided
         * `solveIk()` member function. The kinematic chain used by the solver is defined through
         * the names of its root and tip links, which must exist within the provided URDF
         * description.
         *
         * For more details on the IK solver, refer to the OROCOS KDL
         * project: https://github.com/orocos/orocos_kinematics_dynamics.
         */
        class IkSolverVel_wlds
        {
        public:
            struct IkSolverParams
            {
                double eps;                // Singular value threshold for solver
                double lambda;             // Lambda value for weighted DLS solver
                int max_iters;             // Maximum iterations for the SVD calculation
                Eigen::MatrixXd weight_js; // Joint space weighting symmetric matrix for WDLS
            };

            IkSolverVel_wlds();
            IkSolverVel_wlds(const bool verbose, const std::string &chain_root,
                             const std::string &chain_tip, const IkSolverParams &solver_params);
            IkSolverVel_wlds(const bool verbose, const std::string &chain_root,
                             const std::string &chain_tip, const double eps, const double lambda,
                             const int max_iters, const Eigen::MatrixXd &weight_js);
            IkSolverVel_wlds(const IkSolverVel_wlds &solver);

            IkSolverVel_wlds &operator=(const IkSolverVel_wlds &other);

            bool isInitialized() const;
            int getNumJoints() const;
            void setVerbose(const bool verbose);
            void setChainRoot(const std::string &chain_root);
            void setChainTip(const std::string &chain_tip);
            void setSolverParams(const IkSolverParams &solver_params);
            void initIkSolver(const std::string &urdf_description);
            /**
             * @brief Solve inverse velocity kinematics for the given twist vector at the given
             * joint configuration.
             *
             * @param[in] q Current joint configuration for solving inverse velocity kinematics.
             * @param[in] v Desired twist vector in the task space.
             * @param[out] qdot Output joint velocities computed by the IK solver. Set to zeros if
             * the solver fails.
             */
            void solveIk(const KDL::JntArray &q, const KDL::Twist &v, KDL::JntArray &qdot) const;

        private:
            // General Attributes
            bool m_is_init;           // Whether the solver has been initialized yet or not
            bool m_verbose;           // Enable verbose output
            std::string m_chain_root; // Name of the base frame for inverse kinematics
            std::string m_chain_tip;  // Name of the target frame for inverse kinematics
            KDL::Tree m_tree;         // Kinematic tree to load from URDF robot description
            KDL::Chain m_chain;       // Kinematic chain to solve inverse velocity kinematics over

            // IK Solver and Parameters
            IkSolverParams m_solver_params; // Parameters for the IK solver
            std::unique_ptr<KDL::ChainIkSolverVel_wdls>
                m_solver; // Solver based on weighted DLS method
        };
    } // namespace solver
} // namespace vc

#endif