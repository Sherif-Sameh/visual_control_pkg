/*
 * Description:
 * Utility structs for grouping data or performing simple common functionalities.
 */

#ifndef VC_UTILS_STRUCTS
#define VC_UTILS_STRUCTS

#include <optional>

#include <visp3/visual_features/vpFeatureThetaU.h>
#include <visp3/visual_features/vpFeatureTranslation.h>

#include "rclcpp/rclcpp.hpp"

namespace utils
{
    namespace structs
    {
        /**
         * @brief Wrapper around any type to bundle a `rclcpp::Time` stamp with it.
         *
         * @tparam T Wrapped type.
         */
        template <typename T>
        struct AnyStamped
        {
            rclcpp::Time m_stamp;
            T m_wrapped;
        };

        /**
         * @brief Exponential moving average (EMA) generic template.
         *
         * The EMA very simply computes the latest value according to the following formula:
         * `new_ema` = `new_val` * `alpha` + `old_ema` * (1 - `alpha`).
         * @tparam Value Value type to compute the EMA of.
         * @tparam Scalar Scalar type to use for the `alpha` parameter of the EMA.
         */
        template <typename Value, typename Scalar>
        struct EMA
        {
            Scalar m_alpha;
            Value m_current;

        public:
            void update(const Value &new_value)
            {
                m_current = new_value * m_alpha + m_current * (1 - m_alpha);
            }
        };

        /**
         * @brief Calculator for the period between sucessive function calls using an exponential
         * moving average (EMA).
         *
         * The period should be updated through the `update()` member function with the latest
         * `rclcpp:Time` stamp. The latest computed period can be retrieved at any time through the
         * `get()` member function.
         * @tparam Scalar Scalar type to use for the period and EMA.
         */
        template <typename Scalar>
        struct PeriodEMACalculator
        {
            bool m_is_ema_init = false;
            std::optional<rclcpp::Time> m_stamp_prev;
            EMA<Scalar, Scalar> m_period_ema;

        public:
            void update(rclcpp::Time stamp)
            {
                if (m_stamp_prev.has_value())
                {
                    Scalar dt = static_cast<Scalar>((stamp - *m_stamp_prev).seconds());
                    if (m_is_ema_init) // regular updates
                    {
                        m_period_ema.update(dt);
                    }
                    else // first time initialization
                    {
                        m_is_ema_init = true;
                        m_period_ema.m_current = dt;
                    }
                }
                m_stamp_prev = stamp;
            }

            /**
             * @brief Get the latest computed period.
             *
             * @return `std::optional<Scalar>` Latest scalar period. Is `std::nullopt` if `update()`
             * has not yet been called at least once.
             */
            std::optional<Scalar> get()
            {
                if (m_stamp_prev.has_value()) // period is guaranteed to be initialized
                {
                    return m_period_ema.m_current;
                }
                return std::nullopt;
            }
        };

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