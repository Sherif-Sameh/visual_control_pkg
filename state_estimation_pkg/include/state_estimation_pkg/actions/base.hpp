/*
 * Description:
 * Base action model for EKF implementation.
 */

#ifndef ACTION_BASE
#define ACTION_BASE

#include <cassert>

#include <manif/manif.h>
#include <tl/optional.hpp>

#include "traits.hpp"

namespace se
{
    template <class _Group, class _Action>
    class ActionBase
    {
    public:
        static constexpr int DoF = internal::traits<_Action>::DoF;

        using State = typename manif::LieGroupBase<_Group>::LieGroup;
        using Tangent = typename manif::LieGroupBase<_Group>::Tangent;
        using Action = typename internal::traits<_Action>::Action;
        using Jacobian = typename internal::traits<_Action>::Jacobian;

        using OptJacobianRef = tl::optional<Eigen::Ref<Jacobian>>;

    public:
        MANIF_DEFAULT_CONSTRUCTOR(ActionBase)

        /**
         * @brief Compute tangent action from input state and action with optional Jacobian.
         *
         * @param[in] x Current state (Lie group).
         * @param[in] u Latest input to derive tangent action from.
         * @param[out] J_uout_w Optional Jacobian of action model wrt process noise w.
         * @return An element of the Lie group's tangent space to add to the current state.
         */
        auto operator()(const State &x, const Action &u, OptJacobianRef J_uout_w = {}) const
            -> Tangent;

    protected:
        inline _Action &derived() & noexcept { return *static_cast<_Action *>(this); }
        inline const _Action &derived() const & noexcept
        {
            return *static_cast<const _Action *>(this);
        }
    };

    // Definitions
    template <class _Group, class _Action>
    auto ActionBase<_Group, _Action>::operator()(const State &x, const Action &u,
                                                 OptJacobianRef J_uout_w) const -> Tangent
    {
        return derived()(x, u, J_uout_w);
    }
} // namespace se

#endif