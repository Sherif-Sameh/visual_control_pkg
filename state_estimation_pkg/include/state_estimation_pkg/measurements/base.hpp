/*
 * Description:
 * Base measurement model for EKF implementation.
 */

#ifndef MEASUREMENT_BASE
#define MEASUREMENT_BASE

#include <cassert>

#include <manif/manif.h>
#include <tl/optional.hpp>

#include "traits.hpp"

namespace se
{
    template <class _Group, class _Measure>
    class MeasurementBase
    {
    public:
        static constexpr int DoF = internal::traits<_Measure>::DoF;

        using State = typename manif::LieGroupBase<_Group>::LieGroup;
        using Measurement = typename internal::traits<_Measure>::Measurement;
        using DeltaMeasurement = typename internal::traits<_Measure>::DeltaMeasurement;
        using Jacobian = typename internal::traits<_Measure>::Jacobian;

        using OptJacobianRef = tl::optional<Eigen::Ref<Jacobian>>;

    public:
        MANIF_DEFAULT_CONSTRUCTOR(MeasurementBase)

        /**
         * @brief Compute expected measurement from input state with optional Jacobian.
         *
         * @param[in] x Current state (Lie group).
         * @param[out] J_h_x Optional Jacobian of measurement model wrt input state.
         * @return Expected measurement to compare with input measurement within EKF.
         */
        auto operator()(const State &x, OptJacobianRef J_h_x = {}) const -> Measurement;

    protected:
        inline _Measure &derived() & noexcept { return *static_cast<_Measure *>(this); }
        inline const _Measure &derived() const & noexcept
        {
            return *static_cast<const _Measure *>(this);
        }
    };

    // Definitions
    template <class _Group, class _Measure>
    auto MeasurementBase<_Group, _Measure>::operator()(const State &x, OptJacobianRef J_h_x) const
        -> Measurement
    {
        return derived()(x, J_h_x);
    }
} // namespace se

#endif