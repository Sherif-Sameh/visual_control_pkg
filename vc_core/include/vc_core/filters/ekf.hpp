/*
 * Description:
 * Extened Kalman Filter (EKF) implementation using the manif library for Lie groups.
 * For more on manif, check out the GitHub page: https://github.com/artivis/manif
 */

#ifndef VC_SE_EKF
#define VC_SE_EKF

#include <algorithm>
#include <cassert>

#include <manif/manif.h>

#include "vc_core/actions/unit.hpp"
#include "vc_core/measurements/unit.hpp"

namespace se
{
    /**
     * @brief Generic Extended Kalman Filter (EKF) template for arbitrary Lie Groups with arbitrary
     * action and measurement models.
     *
     * The Q and R covariance matrices can be set through their respective setter member functions.
     * Similarly, the initial state and error covariance P can be set through setter member
     * functions. The EKF provides the two member functions `predict()` and `update()` to perform
     * the two main steps of the EKF algorithm. The state and covariance of the EKF can be retrieved
     * at any time through the `getStateAndCovariance()` member function.
     *
     * Note that all process models are constrained to the following structure: `x_k+1` =
     * `x_k`.rplus(f(`x_k`, `u_k`, `w_k`)), where f(`x_k`, `u_k`, `w_k`) is the given action model.
     * @tparam _Group Lie group, a derived class from `manif::LieGroupBase`.
     * @tparam _Action Action model, a derived class from `se::ActionBase`. Defaults to the unit
     * model defined by `se::ActionUnit` class.
     * @tparam _Measure Measurement model, a derived class from `se::MeasurementBase`. Defaults to
     * the unit model defined by the `se::MeasurementUnit` class.
     */
    template <class _Group, class _Action = ActionUnit<_Group>,
              class _Measure = MeasurementUnit<_Group>>
    class EKF
    {
    public:
        static constexpr int xDoF = manif::LieGroupBase<_Group>::DoF;
        static constexpr int wDoF = ActionBase<_Group, _Action>::DoF;
        static constexpr int yDoF = MeasurementBase<_Group, _Measure>::DoF;

        using Scalar = typename manif::LieGroupBase<_Group>::Scalar;
        using State = typename manif::LieGroupBase<_Group>::LieGroup;
        using Covariance = Eigen::Matrix<Scalar, EKF<_Group, _Action, _Measure>::xDoF, xDoF>;

        using Action = typename ActionBase<_Group, _Action>::Action;
        using CovarianceA = Eigen::Matrix<Scalar, wDoF, wDoF>;

        using Measurement = typename MeasurementBase<_Group, _Measure>::Measurement;
        using CovarianceM = Eigen::Matrix<Scalar, yDoF, yDoF>;

    protected:
        using Tangent = typename manif::LieGroupBase<_Group>::Tangent;
        using Jacobian = typename manif::LieGroupBase<_Group>::Jacobian;
        using KalmanGain = Eigen::Matrix<Scalar, xDoF, yDoF>;
        using JacobianA = typename ActionBase<_Group, _Action>::Jacobian;
        using DeltaMeasurement = typename MeasurementBase<_Group, _Measure>::DeltaMeasurement;
        using JacobianM = typename MeasurementBase<_Group, _Measure>::Jacobian;

    protected:
        static constexpr _Action u_default = ActionUnit<_Group>();
        static constexpr _Measure h_default = MeasurementUnit<_Group>();

    public:
        EKF();
        EKF(const Covariance &P0, const CovarianceA &Q, const CovarianceM &R);

        void getStateAndCovariance(State &x, Covariance &P) const;
        auto getProcessCovariance() const -> CovarianceA;
        auto getMeasurementCovariance() const -> CovarianceM;
        void setState(const State &x0);
        void setErrorCovariance(const Covariance &P0);
        void setProcessCovariance(const CovarianceA &Q);
        void setMeasurementCovariance(const CovarianceM &R);

        /**
         * @brief Peform prediction step of the EKF given the latest action `u`.
         *
         * @param[in] u Latest action whose type is determined by the action model.
         * @param[in] u_func Action model instance to use for mapping actions to the tangent space.
         */
        void predict(const Action &u, const _Action &u_func = u_default);
        /**
         * @brief Perform update step of the EKF given the latest measurement `y`.
         *
         * @param[in] y Latest measurement whose type is determined by the measurement model.
         * @param[in] h_func Measurement model instance to use for mapping state to expected
         * measurement.
         */
        void update(const Measurement &y, const _Measure &h_func = h_default);

    protected:
        template <typename _Covariance>
        bool checkCovarianceMatrix(const _Covariance &A);

    protected:
        State m_x;
        Covariance m_P;
        CovarianceA m_Q;
        CovarianceM m_R;
    };

    // Definitions

    template <class _Group, class _Action, class _Measure>
    EKF<_Group, _Action, _Measure>::EKF()
        : m_x(State::Identity()), m_P(Covariance::Zero()), m_Q(CovarianceA::Identity()),
          m_R(CovarianceM::Identity())
    {
    }

    template <class _Group, class _Action, class _Measure>
    EKF<_Group, _Action, _Measure>::EKF(const Covariance &P0, const CovarianceA &Q,
                                        const CovarianceM &R)
        : m_x(State::Identity()), m_P(P0), m_Q(Q), m_R(R)
    {
    }

    template <class _Group, class _Action, class _Measure>
    void EKF<_Group, _Action, _Measure>::getStateAndCovariance(State &x, Covariance &P) const
    {
        x = m_x;
        P = m_P;
    }

    template <class _Group, class _Action, class _Measure>
    auto EKF<_Group, _Action, _Measure>::getProcessCovariance() const -> CovarianceA
    {
        return m_Q;
    }

    template <class _Group, class _Action, class _Measure>
    auto EKF<_Group, _Action, _Measure>::getMeasurementCovariance() const -> CovarianceM
    {
        return m_R;
    }

    template <class _Group, class _Action, class _Measure>
    void EKF<_Group, _Action, _Measure>::setState(const State &x0)
    {
        m_x = x0;
    }

    template <class _Group, class _Action, class _Measure>
    void EKF<_Group, _Action, _Measure>::setErrorCovariance(const Covariance &P0)
    {
        assert(checkCovarianceMatrix(P0));
        m_P = P0;
    }

    template <class _Group, class _Action, class _Measure>
    void EKF<_Group, _Action, _Measure>::setProcessCovariance(const CovarianceA &Q)
    {
        assert(checkCovarianceMatrix(Q));
        m_Q = Q;
    }

    template <class _Group, class _Action, class _Measure>
    void EKF<_Group, _Action, _Measure>::setMeasurementCovariance(const CovarianceM &R)
    {
        assert(checkCovarianceMatrix(R));
        m_R = R;
    }

    template <class _Group, class _Action, class _Measure>
    void EKF<_Group, _Action, _Measure>::predict(const Action &u, const _Action &u_func)
    {
        JacobianA J_uw;
        Tangent tan = u_func(m_x, u, J_uw);

        Jacobian J_fx, J_fu;
        m_x = m_x.plus(tan, J_fx, J_fu);
        m_P =
            J_fx * m_P * J_fx.transpose() + J_fu * J_uw * m_Q * J_uw.transpose() * J_fu.transpose();
    }

    template <class _Group, class _Action, class _Measure>
    void EKF<_Group, _Action, _Measure>::update(const Measurement &y, const _Measure &h_func)
    {
        JacobianM J_hx;
        Measurement y_exp = h_func(m_x, J_hx);

        DeltaMeasurement z = y - y_exp;
        CovarianceM Z = J_hx * m_P * J_hx.transpose() + m_R;
        KalmanGain K = m_P * J_hx.transpose() * Z.inverse();
        m_x = m_x.plus(K * z);
        m_P = m_P - K * Z * K.transpose();
    }

    template <class _Group, class _Action, class _Measure>
    template <typename _Covariance>
    bool EKF<_Group, _Action, _Measure>::checkCovarianceMatrix(const _Covariance &A)
    {
        assert(A.isApprox(A.transpose()));
        assert(
            std::all_of(A.diagonal().begin(), A.diagonal().end(), [](Scalar x) { return x >= 0; }));
        return true;
    }
} // namespace se

#endif