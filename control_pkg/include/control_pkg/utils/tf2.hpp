/*
 * Description:
 * Utility functions related to ROS TF2.
 */

#ifndef CONTROL_UTILS_TF2
#define CONTROL_UTILS_TF2

#include <memory>

#include <visp3/core/vpHomogeneousMatrix.h>

#include "tf2/exceptions.hpp"
#include "tf2_ros/buffer.hpp"

#include "utils/conversions/geometry.hpp"

namespace utils
{
    namespace ros_tf2
    {
        /**
         * @brief Lookup transform from TF tree between target and source frames.
         *
         * @param[in] target_frame Target frame for desired transform.
         * @param[in] source_frame Source frame for desired transform.
         * @param[in] buffer TF2 buffer to use for frame lookup.
         * @param[out] t Output transformation if lookup was successful.
         * @return true If lookup was successful and transform was updated.
         * @return false If lookup failed and a `tf2::TransformException` was raised.
         */
        inline bool lookup_transform(const std::string &target_frame,
                                     const std::string &source_frame,
                                     const std::unique_ptr<tf2_ros::Buffer> &buffer,
                                     vpHomogeneousMatrix &t)
        {
            geometry_msgs::msg::TransformStamped t_gm;
            try
            {
                t_gm = buffer->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
            }
            catch (const tf2::TransformException &ex)
            {
                return false;
            }
            t = geometry::gm_transform_to_vp_hmatrix(t_gm.transform);
            return true;
        }
    } // namespace ros_tf2
} // namespace utils

#endif