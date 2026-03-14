/*
 * Description:
 * Unit action model for EKF implementation.
 */

#ifndef ACTION_UNIT
#define ACTION_UNIT

#include "vc_core/actions/base.hpp"
#include "vc_core/traits.hpp"

namespace se
{
    /**
     * @brief Unit action model.
     *
     * An action model representing the identity action to tangent mapping. It does not modify its
     * input actions at all. Input actions are returned as they are and therefore, its Jacobian, if
     * required, is the identity matrix.
     *
     * Accordingly, its input action type is the same as that of the associated Lie Group's tangent
     * space. The number of DoF of the process noise and the size of its Jacobian are derived from
     * the number of DoF of its associated Lie Group.
     * @tparam _Group Lie group, a derived class from `manif::LieGroupBase`.
     */
    template <class _Group>
    class ActionUnit : public ActionBase<_Group, ActionUnit<_Group>>
    {
    public:
        using Base = ActionBase<_Group, ActionUnit<_Group>>;

        // bring from base into derived scope
        using State = typename Base::State;
        using Tangent = typename Base::Tangent;
        using Action = typename Base::Action;
        using Jacobian = typename Base::Jacobian;

        using OptJacobianRef = typename Base::OptJacobianRef;

    public:
        /**
         * @brief Compute tangent action from input state and action with optional Jacobian.
         *
         * Returns the input action without any modifications. The Jacobian is therefore the
         * identity.
         * @param[in] x Current state (Lie group).
         * @param[in] u Latest input to derive tangent action from.
         * @param[out] J_uout_w Optional Jacobian of action model wrt process noise `w`.
         * @return An element of the Lie group's tangent space to add to the current state.
         */
        auto operator()(const State &x, const Action &u, OptJacobianRef J_uout_w = {}) const
            -> Tangent;
    };

    // Definitions

    template <class _Group>
    auto ActionUnit<_Group>::operator()(const State &x, const Action &u,
                                        OptJacobianRef J_uout_w) const -> Tangent
    {
        (void)(x);
        if (J_uout_w)
        {
            (*J_uout_w) = Jacobian::Identity();
        }
        return u;
    }

    // Internal traits definition
    namespace internal
    {
        template <class _Group>
        struct traits<ActionUnit<_Group>>
        {
            static constexpr int DoF = manif::LieGroupBase<_Group>::DoF;

            using Scalar = typename manif::LieGroupBase<_Group>::Scalar;
            using Action = typename manif::LieGroupBase<_Group>::Tangent;
            using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
        };
    } // namespace internal
} // namespace se

#endif