#include "kdlIkSolVel_wdls.hpp"

namespace vc
{
    namespace ik
    {
        kdlIkSolVel_wlds::kdlIkSolVel_wlds(const bool verbose, const std::string &chain_root,
                                           const std::string &chain_tip, const double eps,
                                           const int max_iters, const double lambda,
                                           const Eigen::MatrixXd &weight_js)
            : m_is_init(false), m_verbose(verbose), m_chain_root(chain_root),
              m_chain_tip(chain_tip), m_solver_eps(eps), m_solver_max_iters(max_iters),
              m_solver_lambda(lambda), m_solver_weight_js(weight_js)
        {
            assert(m_solver_weight_js.rows() == m_solver_weight_js.cols());
        }

        void kdlIkSolVel_wlds::initIkSolver(const std::string &urdf_description)
        {
            // Construct KDL tree from URDF description string
            kdl_parser::treeFromString(urdf_description, m_tree);
            if (m_verbose)
            {
                // Print basic information about the tree
                std::cout << "Kinematic Tree Information:" << std::endl;
                std::cout << "nb joints:        " << m_tree.getNrOfJoints() << std::endl;
                std::cout << "nb segments:      " << m_tree.getNrOfSegments() << std::endl;
                std::cout << "root segment:     " << m_tree.getRootSegment()->first << std::endl;
            }

            // Extract target kinematic chain from tree
            m_tree.getChain(m_chain_root, m_chain_tip, m_chain);
            assert(m_solver_weight_js.rows() == m_chain.getNrOfJoints());
            if (m_verbose)
            {
                std::cout << "Kinematic Chain Information:" << std::endl;
                std::cout << "nb joints:        " << m_chain.getNrOfJoints() << std::endl;
                std::cout << "nb segments:      " << m_chain.getNrOfSegments() << std::endl;
                std::cout << "root segment:     " << m_chain_root << std::endl;
                std::cout << "tip segment:      " << m_chain_tip << std::endl;
            }

            // Initialize solver
            m_is_init = true;
            m_solver = std::make_unique<KDL::ChainIkSolverVel_wdls>(m_chain, m_solver_eps,
                                                                    m_solver_max_iters);
            m_solver->setLambda(m_solver_lambda);
            m_solver->setWeightJS(m_solver_weight_js);
            if (m_verbose)
            {
                std::cout << "Inverse Velocity Kinematics Solver (WDLS) Initialized" << std::endl;
                std::cout << "IK Solver Information:" << std::endl;
                std::cout << "eps:              " << m_solver_eps << std::endl;
                std::cout << "max_iters:        " << m_solver_max_iters << std::endl;
                std::cout << "lambda:           " << m_solver_lambda << std::endl;
                std::cout << "weight_js:        " << m_solver_weight_js << std::endl;
            }
        }

        int kdlIkSolVel_wlds::getNumJoints() const
        {
            assert(m_is_init);
            if (m_is_init)
            {
                return m_chain.getNrOfJoints();
            }
            return 0;
        }

        void kdlIkSolVel_wlds::solveIk(const std::vector<double> &q, const std::vector<double> &v,
                                       std::vector<double> &qdot)
        {
            assert(m_is_init);
            assert(q.size() == m_chain.getNrOfJoints());
            assert(v.size() == 6);
            assert(q.size() == qdot.size());

            if (m_is_init)
            {
                // Initialize KDL arrays from inputs
                KDL::JntArray q_array(m_chain.getNrOfJoints()), qdot_array(m_chain.getNrOfJoints());
                q_array.data =
                    Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(q.data(), q.size());
                qdot_array.data =
                    Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(qdot.data(), qdot.size());
                KDL::Twist v_array(KDL::Vector(v[0], v[1], v[2]), KDL::Vector(v[3], v[4], v[5]));

                // Invoke solver with inputs
                int error = m_solver->CartToJnt(q_array, v_array, qdot_array);
                if (error != KDL::SolverI::E_NOERROR)
                {
                    // Ignore any computed velocities in case of failure
                    std::fill(qdot.begin(), qdot.end(), 0.0);
                    std::cout << "Solver failed due to error number: " << error << std::endl;
                }
            }
        }
    } // namespace ik
} // namespace vc