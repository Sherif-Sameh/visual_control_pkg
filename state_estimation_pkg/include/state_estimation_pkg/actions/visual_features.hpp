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
    /**
     * @brief Base visual features action model.
     *
     * An action model representing the action to tangent mapping for arbitrary visual features.
     * Input actions represent tangent vectors defined at the observing camera's pose on the SE3
     * manifold. Input actions are mapped to the feature tangent space through the feature's
     * interaction matrix. It's assumed that process noise `w` is indepenent of this mapping and
     * therefore its Jacobian is the identity.
     *
     * Derived action models are expected to define the size of their interaction matrices through
     * specializations of the `internal::traits` struct. They are also expected to provide the
     * implementation of the `interaction()` function for computing their interaction matrix at the
     * current state.
     * @tparam _Group Lie group, a derived class from `manif::LieGroupBase`.
     * @tparam _Feature Derived feature action model class for CRTP. For more on CRTP, refer to:
     * https://en.cppreference.com/w/cpp/language/crtp.html
     */
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

        using Interaction = typename internal::traits<_Feature>::Interaction;

        using OptJacobianRef = typename Base::OptJacobianRef;

    public:
        MANIF_DEFAULT_CONSTRUCTOR(ActionVisualFeatures)

        /**
         * @brief Compute tangent action from input state and action with optional Jacobian.
         *
         * Input actions are camera twist vectors. Process noise is assumed to be independent of
         * camera velocity and therefore the Jacobian is the identity.
         * @param[in] x Current state (Lie group).
         * @param[in] u Latest input to derive tangent action from.
         * @param[out] J_uout_w Optional Jacobian of action model wrt process noise `w`.
         * @return An element of the Lie group's tangent space to add to the current state.
         */
        auto operator()(const State &x, const Action &u, OptJacobianRef J_uout_w = {}) const
            -> Tangent;

    protected:
        using Base::derived;

        /**
         * @brief Compute and return the feature's interaction matrix at the current state.
         *
         * @param[in] x Current state (Lie Group).
         * @return Interaction matrix evaluated at the current state.
         */
        auto interaction(const State &x) const -> Interaction;
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
        return interaction() * u;
    }

    template <class _Group, class _Feature>
    auto ActionVisualFeatures<_Group, _Feature>::interaction(const State &x) const -> Interaction
    {
        return derived().interaction();
    }
} // namespace se

#endif