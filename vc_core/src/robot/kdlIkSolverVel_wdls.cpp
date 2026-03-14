#include "vc_core/robot/kdlIkSolverVel_wdls.hpp"

namespace vc
{
    namespace solver
    {
        KdlIkSolverVel_wlds::KdlIkSolverVel_wlds()
            : m_is_init(false), m_verbose(false), m_chain_root("root"), m_chain_tip("tip"),
              m_solver_params{1e-5, 0.0, 50, Eigen::MatrixXd::Identity(6, 6)}, m_solver(nullptr)
        {
        }

        KdlIkSolverVel_wlds::KdlIkSolverVel_wlds(const bool verbose, const std::string &chain_root,
                                                 const std::string &chain_tip,
                                                 const IkSolverParams &solver_params)
            : m_is_init(false), m_verbose(verbose), m_chain_root(chain_root),
              m_chain_tip(chain_tip), m_solver_params(solver_params), m_solver(nullptr)
        {
            if (m_solver_params.weight_js.rows() != m_solver_params.weight_js.cols())
            {
                throw(vpException(vpException::dimensionError,
                                  "Solver weight matrix for joint states must be square."));
            }
        }

        KdlIkSolverVel_wlds::KdlIkSolverVel_wlds(const bool verbose, const std::string &chain_root,
                                                 const std::string &chain_tip, const double eps,
                                                 const double lambda, const int max_iters,
                                                 const Eigen::MatrixXd &weight_js)
            : KdlIkSolverVel_wlds(verbose, chain_root, chain_tip,
                                  {eps, lambda, max_iters, weight_js})
        {
        }

        KdlIkSolverVel_wlds::KdlIkSolverVel_wlds(const KdlIkSolverVel_wlds &solver)
            : KdlIkSolverVel_wlds(solver.m_verbose, solver.m_chain_root, solver.m_chain_tip,
                                  solver.m_solver_params)
        {
        }

        KdlIkSolverVel_wlds &KdlIkSolverVel_wlds::operator=(const KdlIkSolverVel_wlds &other)
        {
            m_is_init = false; // must be re-initialized due to unique_ptr
            m_verbose = other.m_verbose;
            m_chain_root = other.m_chain_root;
            m_chain_tip = other.m_chain_tip;
            m_solver_params = other.m_solver_params;
            m_solver = nullptr;
            return *this;
        }

        bool KdlIkSolverVel_wlds::isInitialized() const { return m_is_init; }

        int KdlIkSolverVel_wlds::getNumJoints() const
        {
            if (!m_is_init)
            {
                throw(vpException(vpException::notInitialized,
                                  "Solver has not yet been initialized."));
            }
            return m_chain.getNrOfJoints();
        }

        void KdlIkSolverVel_wlds::setVerbose(const bool verbose) { m_verbose = verbose; }

        void KdlIkSolverVel_wlds::setChainRoot(const std::string &chain_root)
        {
            m_chain_root = chain_root;
        }

        void KdlIkSolverVel_wlds::setChainTip(const std::string &chain_tip)
        {
            m_chain_tip = chain_tip;
        }

        void KdlIkSolverVel_wlds::setSolverParams(const IkSolverParams &solver_params)
        {
            m_solver_params = solver_params;
        }

        void KdlIkSolverVel_wlds::initIkSolver(const std::string &urdf_description)
        {
            // Construct KDL tree from URDF description string
            kdl_parser::treeFromString(urdf_description, m_tree);
            if (m_verbose)
            {
                // Print basic information about the tree
                std::cout << "Kinematic Tree Information:\n";
                std::cout << "nb joints:        " << m_tree.getNrOfJoints() << "\n";
                std::cout << "nb segments:      " << m_tree.getNrOfSegments() << "\n";
                std::cout << "root segment:     " << m_tree.getRootSegment()->first << "\n";
                std::cout << std::endl;
            }

            // Extract target kinematic chain from tree
            m_tree.getChain(m_chain_root, m_chain_tip, m_chain);
            if (m_solver_params.weight_js.rows() != m_chain.getNrOfJoints())
            {
                throw(vpException(vpException::dimensionError,
                                  "Dimension of solver weight matrix for joint states does not "
                                  "match number of joints in kinematic chain."));
            }
            if (m_verbose)
            {
                std::cout << "Kinematic Chain Information:\n";
                std::cout << "nb joints:        " << m_chain.getNrOfJoints() << "\n";
                std::cout << "nb segments:      " << m_chain.getNrOfSegments() << "\n";
                std::cout << "root segment:     " << m_chain_root << "\n";
                std::cout << "tip segment:      " << m_chain_tip << "\n";
                std::cout << std::endl;
            }

            // Initialize solver
            m_is_init = true;
            m_solver = std::make_unique<KDL::ChainIkSolverVel_wdls>(m_chain, m_solver_params.eps,
                                                                    m_solver_params.max_iters);
            m_solver->setLambda(m_solver_params.lambda);
            m_solver->setWeightJS(m_solver_params.weight_js);
            if (m_verbose)
            {
                std::cout << "Inverse Velocity Kinematics Solver (WDLS) Initialized.\n";
                std::cout << "IK Solver Information:\n";
                std::cout << "eps:              " << m_solver_params.eps << "\n";
                std::cout << "max_iters:        " << m_solver_params.max_iters << "\n";
                std::cout << "lambda:           " << m_solver_params.lambda << "\n";
                std::cout << "weight_js:        " << "\n" << m_solver_params.weight_js << "\n";
                std::cout << std::endl;
            }
        }

        void KdlIkSolverVel_wlds::solveIk(const KDL::JntArray &q, const KDL::Twist &v,
                                          KDL::JntArray &qdot) const
        {
            assert(m_is_init);
            assert(q.data.size() == m_chain.getNrOfJoints());
            assert(q.data.size() == qdot.data.size());

            // Invoke solver with inputs
            int error = m_solver->CartToJnt(q, v, qdot);
            if (error != KDL::SolverI::E_NOERROR)
            {
                // Ignore any computed velocities in case of failure
                KDL::SetToZero(qdot);
                if (m_verbose)
                {
                    std::cerr << "Failure: Solver failed due to error: "
                              << m_solver->strError(error) << std::endl;
                }
            }
        }
    } // namespace solver
} // namespace vc