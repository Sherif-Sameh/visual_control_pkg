/*
 * Description:
 * SE3 visual features action model for EKF implementation.
 */

#ifndef ACTION_SE3_FEATURES
#define ACTION_SE3_FEATURES

#include "actions/visual_features.hpp"
#include "traits.hpp"

namespace se
{
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

    protected:
        auto interaction(const State &x) const -> Interaction;
    };

    // Definitions

    template <typename _Scalar>
    auto ActionSE3Features<_Scalar>::interaction(const State &x) const -> Interaction
    {
        return x.inverse().adj();
    }
} // namespace se

#endif