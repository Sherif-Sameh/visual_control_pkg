/*
 * Description:
 * Wrapper around the ChainIkSolverVel_wdls inverse velocity kinematics solver from OROCOS KDL.
 * For more, check out the OROCOS KDL project: https://github.com/orocos/orocos_kinematics_dynamics
 */

#ifndef KDL_IK_SOLVER_VEL_WDLS
#define KDL_IK_SOLVER_VEL_WDLS

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
        class KdlIkSolverVel_wlds
        {
        public:
            struct IkSolverParams
            {
                double eps;                // Singular value threshold for solver
                double lambda;             // Lambda value for weighted DLS solver
                int max_iters;             // Maximum iterations for the SVD calculation
                Eigen::MatrixXd weight_js; // Joint space weighting symmetric matrix for WDLS
            };

            KdlIkSolverVel_wlds(const bool verbose, const std::string &chain_root,
                                const std::string &chain_tip, const IkSolverParams &solver_params);
            KdlIkSolverVel_wlds(const bool verbose, const std::string &chain_root,
                                const std::string &chain_tip, const double eps, const double lambda,
                                const int max_iters, const Eigen::MatrixXd &weight_js);
            KdlIkSolverVel_wlds(const KdlIkSolverVel_wlds &solver);
            bool isInitialized() const;
            int getNumJoints() const;
            void initIkSolver(const std::string &urdf_description);
            void solveIk(const KDL::JntArray &q, const KDL::Twist &v, KDL::JntArray &qdot) const;

        private:
            // General Attributes
            bool m_is_init;                 // Whether the solver has been initialized yet or not
            const bool m_verbose;           // Enable verbose output
            const std::string m_chain_root; // Name of the base frame for inverse kinematics
            const std::string m_chain_tip;  // Name of the target frame for inverse kinematics
            KDL::Tree m_tree;               // Kinematic tree to load from URDF robot description
            KDL::Chain m_chain; // Kinematic chain to solve inverse velocity kinematics over

            // IK Solver and Parameters
            const IkSolverParams m_solver_params; // Parameters for the IK solver
            std::unique_ptr<KDL::ChainIkSolverVel_wdls>
                m_solver; // Solver based on weighted DLS method
        };
    } // namespace solver
} // namespace vc

#endif