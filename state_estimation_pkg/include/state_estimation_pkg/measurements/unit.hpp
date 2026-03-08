/*
 * Description:
 * Unit measurement model for EKF implementation.
 */

#ifndef MEASUREMENT_UNIT
#define MEASUREMENT_UNIT

#include "measurements/base.hpp"

namespace se
{
    template <class _Group>
    class MeasurementUnit : public MeasurementBase<_Group, MeasurementUnit<_Group>>
    {
    public:
        using Base = MeasurementBase<_Group, MeasurementUnit<_Group>>;

        // bring from base into derived scope
        using State = typename Base::State;
        using Measurement = typename Base::Measurement;
        using DeltaMeasurement = typename Base::DeltaMeasurement;
        using Jacobian = typename Base::Jacobian;
        using OptJacobianRef = typename Base::OptJacobianRef;

    public:
        /**
         * @brief Compute expected measurement from input state with optional Jacobian.
         *
         * Returns  the input state without any modifications. The Jacobian is therefore the
         * identity.
         * @param[in] x Current state (Lie group).
         * @param[out] J_h_x Optional Jacobian of measurement model wrt input state.
         * @return Expected measurement to compare with input measurement within EKF.
         */
        auto operator()(const State &x, OptJacobianRef J_h_x = {}) const -> Measurement;
    };

    // Definitions

    template <class _Group>
    auto MeasurementUnit<_Group>::operator()(const State &x, OptJacobianRef J_h_x) const
        -> Measurement
    {
        if (J_h_x)
        {
            (*J_h_x) = Jacobian::Identity();
        }
        return x;
    }

    // Internal traits definition
    namespace internal
    {
        template <class _Group>
        struct traits<MeasurementUnit<_Group>>
        {
            static constexpr int DoF = manif::LieGroupBase<_Group>::DoF;

            using Scalar = typename manif::LieGroupBase<_Group>::Scalar;
            using Measurement = typename manif::LieGroupBase<_Group>::LieGroup;
            using DeltaMeasurement = typename manif::LieGroupBase<_Group>::Tangent;
            using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;
        };
    } // namespace internal
} // namespace se

#endif