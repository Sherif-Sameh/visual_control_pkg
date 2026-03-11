/*
 * Description:
 * Utility structs for grouping data or performing simple common functionalities.
 */

#ifndef CONTROL_UTILS_STRUCTS
#define CONTROL_UTILS_STRUCTS

#include <visp3/visual_features/vpFeatureThetaU.h>
#include <visp3/visual_features/vpFeatureTranslation.h>

namespace utils
{
    namespace structs
    {
        /**
         * @brief Struct grouping visp SE3 pose features together.
         *
         * Groups both a `vpFeatureTranslation` and a `vpFeatureThetaU` together.
         */
        struct vpPoseFeature
        {
            vpFeatureTranslation m_t;
            vpFeatureThetaU m_tu;
        };
    } // namespace structs
} // namespace utils

#endif