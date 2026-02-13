/*
 * Description:
 * Utility functions for converting between different geometrical pose representations.
 */

#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpQuaternionVector.h>
#include <visp3/core/vpTranslationVector.h>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform.hpp>

/**
 * @brief Convert pose from geometry_msgs Pose to ViSP vpHomogeneousMatrix
 *
 * @param pose Input pose to initialize vpHomogeneousMatrix from.
 * @return vpHomogeneousMatrix
 */
vpHomogeneousMatrix gm_pose_to_vp_hmatrix(const geometry_msgs::msg::Pose &pose)
{
    return vpHomogeneousMatrix(
        vpTranslationVector(pose.position.x, pose.position.y, pose.position.z),
        vpQuaternionVector(pose.orientation.x, pose.orientation.y, pose.orientation.z,
                           pose.orientation.w));
}

/**
 * @brief Convert pose from geometry_msgs Transform to ViSP vpHomogeneousMatrix
 *
 * @param transform Input transform to initialize vpHomogeneousMatrix from.
 * @return vpHomogeneousMatrix
 */
vpHomogeneousMatrix gm_transform_to_vp_hmatrix(const geometry_msgs::msg::Transform &transform)
{
    return vpHomogeneousMatrix(vpTranslationVector(transform.translation.x, transform.translation.y,
                                                   transform.translation.z),
                               vpQuaternionVector(transform.rotation.x, transform.rotation.y,
                                                  transform.rotation.z, transform.rotation.w));
}
