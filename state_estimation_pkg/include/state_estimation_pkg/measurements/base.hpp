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
    /**
     * @brief Base measurement model.
     *
     * Measurement models are responsible for defining the mapping from the current state into the
     * expected measurements for any arbitrary measurement representation. Additionally, they should
     * provide the Jacobian of the measurement function wrt the input state. A measurement model is
     * called through its `operator ()`.
     *
     * Derived measurement models are expected to define the types of both output measurements,
     * measurement residual (i.e., `y - y_exp`) and Jacobian as well as the number of DoF of
     * measurements through specializations of the `internal::traits` struct.
     * @tparam _Group Lie group, a derived class from `manif::LieGroupBase`.
     * @tparam _Measure Derived measurement model class for CRTP. For more on CRTP, refer to:
     * https://en.cppreference.com/w/cpp/language/crtp.html
     */
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