/*
 * Description:
 * SE3 visual features action model for EKF implementation.
 */

#ifndef ACTION_SE3_FEATURES
#define ACTION_SE3_FEATURES

#include "vc_core/actions/visual_features.hpp"
#include "vc_core/traits.hpp"

namespace se
{
    /**
     * @brief SE3 visual features action model.
     *
     * An action model for visual features in SE3. It provides the mapping from input
     * actions represented in the observing camera's tangent space to the tangent space of the
     * visual features. Since both are in SE3, this only requires a change of basis for the input.
     * This is achieved through the negative of the Adjoint mapping of the inverse of the current
     * state of the visual features.
     *
     * @tparam _Scalar Scalar type of the visual features (e.g., `double`).
     */
    template <typename _Scalar>
    class ActionSE3Features
        : public ActionVisualFeatures<manif::SE3<_Scalar>, ActionSE3Features<_Scalar>>
    {
    public:
        using Base = ActionVisualFeatures<manif::SE3<_Scalar>, ActionSE3Features<_Scalar>>;

        // bring from base into derived scope
        using State = typename Base::State;
        using Tangent = typename Base::Tangent;
        using Action = typename Base::Action;
        using Jacobian = typename Base::Jacobian;
        using Interaction = typename Base::Interaction;

        using OptJacobianRef = typename Base::OptJacobianRef;

    public:
        /**
         * @brief Compute and return the feature's interaction matrix at the current state.
         *
         * @param[in] x Current state (Lie Group).
         * @return Interaction matrix evaluated at the current state.
         */
        auto interaction(const State &x) const -> Interaction;
    };

    // Definitions

    template <typename _Scalar>
    auto ActionSE3Features<_Scalar>::interaction(const State &x) const -> Interaction
    {
        return -x.inverse().adj();
    }

    // Internal traits definition
    namespace internal
    {
        template <typename _Scalar>
        struct traits<ActionSE3Features<_Scalar>>
        {
            static constexpr int DoF = manif::SE3<_Scalar>::DoF;

            using Scalar = _Scalar;
            using Action = Eigen::Matrix<_Scalar, 6, 1>;
            using Jacobian = Eigen::Matrix<_Scalar, DoF, DoF>;
            using Interaction = Eigen::Matrix<_Scalar, DoF, 6>;
        };
    } // namespace internal
} // namespace se

#endif