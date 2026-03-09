/*
 * Description:
 * 2D image point visual features action model for EKF implementation.
 */

#ifndef ACTION_POINT_FEATURES
#define ACTION_POINT_FEATURES

#include <array>

#include "actions/visual_features.hpp"
#include "traits.hpp"

namespace se
{
    /**
     * @brief 2D image point(s) visual features action model.
     *
     * An action model for 2D image point visual features. It provides the mapping from input
     * actions represented in the observing camera's tangent space to 2D feature displacements
     * expressed in R(2N), where N is the number of points.
     *
     * Prior to calling this action model, the Z (depth)-coordinates of each point should be
     * provided through the `setZ()` member function since they're required in the computation of
     * the interaction matrix.
     * @tparam _Scalar Scalar type of the visual features (e.g., `double`).
     * @tparam NumPts Number of 2D image point visual features.
     */
    template <typename _Scalar, unsigned int NumPts>
    class ActionPointFeatures : public ActionVisualFeatures<manif::Rn<_Scalar, 2 * NumPts>,
                                                            ActionPointFeatures<_Scalar, NumPts>>
    {
    public:
        using Base = ActionVisualFeatures<manif::Rn<_Scalar, 2 * NumPts>,
                                          ActionPointFeatures<_Scalar, NumPts>>;

        // bring from base into derived scope
        using State = typename Base::State;
        using Tangent = typename Base::Tangent;
        using Action = typename Base::Action;
        using Jacobian = typename Base::Jacobian;
        using Interaction = typename Base::Interaction;

        using OptJacobianRef = typename Base::OptJacobianRef;

    public:
        ActionPointFeatures() { m_z.fill(static_cast<_Scalar>(1)); }
        void setZ(const std::array<_Scalar, NumPts> &z) { m_z = z; }
        void setZ(const std::array<_Scalar, NumPts> &&z) { m_z = std::move(z); }

    protected:
        /**
         * @brief Compute and return the feature's interaction matrix at the current state.
         *
         * @param[in] x Current state (Lie Group).
         * @return Interaction matrix evaluated at the current state.
         */
        auto interaction(const State &x) const -> Interaction;

    protected:
        std::array<_Scalar, NumPts> m_z;
    };

    // Definitions

    template <typename _Scalar, unsigned int NumPts>
    auto ActionPointFeatures<_Scalar, NumPts>::interaction(const State &x) const -> Interaction
    {
        Interaction L;
        for (unsigned int i = 0; i < NumPts; i++)
        {
            const _Scalar xi = x[2 * i], yi = x[2 * i + 1], zi = m_z[i];
            L.row(2 * i) << -1 / zi, 0, xi / zi, xi * yi, -(1 + xi * xi), yi;
            L.row(2 * 1 + 1) << 0, -1 / zi, yi / zi, 1 + yi * yi, -xi * yi, -xi;
        }
        return L;
    }

    // Internal traits definition
    namespace internal
    {
        template <typename _Scalar, unsigned int NumPts>
        struct traits<ActionPointFeatures<_Scalar, NumPts>>
        {
            static constexpr int DoF = manif::Rn<_Scalar, 2 * NumPts>::DoF;

            using Scalar = _Scalar;
            using Action = Eigen::Matrix<_Scalar, 6, 1>;
            using Jacobian = Eigen::Matrix<_Scalar, DoF, DoF>;
            using Interaction = Eigen::Matrix<_Scalar, DoF, 6>;
        };
    } // namespace internal
} // namespace se

#endif