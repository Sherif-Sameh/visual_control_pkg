/*
 * Description:
 * Base visual features action model for EKF implementation.
 */

#ifndef ACTION_VISUAL_FEATURES
#define ACTION_VISUAL_FEATURES

#include "actions/base.hpp"
#include "traits.hpp"

namespace se
{
    template <class _Group, class _Feature>
    class ActionVisualFeatures : public ActionBase<_Group, _Feature>
    {
    public:
        using Base = ActionBase<_Group, _Feature>;

        // bring from base into derived scope
        using State = typename Base::State;
        using Tangent = typename Base::Tangent;
        using Action = typename Base::Action;
        using Jacobian = typename Base::Jacobian;

        using Scalar = typename internal::traits<_Feature>::Scalar;
        using Interaction = typename internal::traits<_Feature>::Interaction;

        using OptJacobianRef = typename Base::OptJacobianRef;

    public:
        MANIF_DEFAULT_CONSTRUCTOR(ActionVisualFeatures)

        void setDt(const Scalar dt) { m_dt = dt; }

        /**
         * @brief Compute tangent action from input state and action with optional Jacobian.
         *
         * Input actions are camera twist vectors. Process noise is assumed to be independent of
         * camera velocity and therefore the Jacobian is the identity.
         * @param[in] x Current state (Lie group).
         * @param[in] u Latest input to derive tangent action from.
         * @param[out] J_uout_w Optional Jacobian of action model wrt process noise w.
         * @return An element of the Lie group's tangent space to add to the current state.
         */
        auto operator()(const State &x, const Action &u, OptJacobianRef J_uout_w = {}) const
            -> Tangent;

    protected:
        using Base::derived;

        auto interaction(const State &x) const -> Interaction;

    protected:
        Scalar m_dt;
    };

    // Definitions

    template <class _Group, class _Feature>
    auto ActionVisualFeatures<_Group, _Feature>::operator()(const State &x, const Action &u,
                                                            OptJacobianRef J_uout_w) const
        -> Tangent
    {
        if (J_uout_w)
        {
            (*J_uout_w) = Jacobian::Identity();
        }
        return m_dt * interaction() * u;
    }

    template <class _Group, class _Feature>
    auto ActionVisualFeatures<_Group, _Feature>::interaction(const State &x) const -> Interaction
    {
        return derived().interaction();
    }

    // Internal traits definition
    namespace internal
    {
        template <class _Group, class _Feature>
        struct traits<ActionVisualFeatures<_Group, _Feature>>
        {
            static constexpr int DoF = manif::LieGroupBase<_Group>::DoF;

            using Scalar = typename manif::LieGroupBase<_Group>::Scalar;
            using Action = Eigen::Matrix<Scalar, 6, 1>;
            using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
            using Interaction = Eigen::Matrix<Scalar, DoF, 6>;
        };
    } // namespace internal
} // namespace se

#endif