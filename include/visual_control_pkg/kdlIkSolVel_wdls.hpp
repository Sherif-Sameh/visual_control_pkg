/*
 * Description:
 * Wrapper around the ChainIkSolverVel_wdls inverse velocity kinematics solver from OROCOS KDL.
 * For more, check out the OROCOS KDL project: https://github.com/orocos/orocos_kinematics_dynamics
 */

#ifndef KDL_IK_SOL_VEL_WDLS
#define KDL_IK_SOL_VEL_WDLS

#include <algorithm>
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

namespace vc
{
    namespace ik
    {
        class kdlIkSolVel_wlds
        {
        public:
            kdlIkSolVel_wlds(const bool verbose, const std::string &chain_root,
                             const std::string &chain_tip, const double eps, const int max_iters,
                             const double lambda, const Eigen::MatrixXd &weight_js);
            void initIkSolver(const std::string &urdf_description);
            int getNumJoints() const;
            void solveIk(const std::vector<double> &q, const std::vector<double> &v,
                         std::vector<double> &qdot);

        private:
            // General Attributes
            bool m_is_init;                 // Whether the solver has been initialized yet or not
            const bool m_verbose;           // Enable verbose output
            const std::string m_chain_root; // Name of the base frame for inverse kinematics
            const std::string m_chain_tip;  // Name of the target frame for inverse kinematics
            KDL::Tree m_tree;               // Kinematic tree to load from URDF robot description
            KDL::Chain m_chain; // Kinematic chain to solve inverse velocity kinematics over

            // IK Solver Attributes
            const double m_solver_eps;    // Singular value threshold for solver
            const int m_solver_max_iters; // Maximum iterations for the SVD calculation
            const double m_solver_lambda; // Lambda value for weighted DLS solver.
            const Eigen::MatrixXd
                m_solver_weight_js; // Joint space weighting symmetric matrix for WDLS
            std::unique_ptr<KDL::ChainIkSolverVel_wdls>
                m_solver; // Solver based on weighted DLS method
        };
    } // namespace ik
} // namespace vc

#endif