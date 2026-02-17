/*
 * Description:
 * Utility functions for converting between different geometrical pose representations.
 */

#include <cassert>
#include <vector>

#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpQuaternionVector.h>
#include <visp3/core/vpTranslationVector.h>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

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

/**
 * @brief Convert rotation from (x, y, z) 3D axis-angle to geometry_msgs Quaternion.
 *
 * @param x x-component of the axis-angle representation.
 * @param y y-component of the axis-angle representation.
 * @param z z-component of the axis-angle representation.
 * @return geometry_msgs::msg::Quaternion
 */
geometry_msgs::msg::Quaternion xyz_aa_to_gm_quat(const double x, const double y, const double z)
{
    constexpr double angle_threshold = 1e-6;

    // Construct tf2 Quaternion from axis-angle representation
    tf2::Quaternion tf2_quat;
    tf2::Vector3 axis(x, y, z);
    double angle = axis.length();
    if (angle < angle_threshold)
    {
        tf2_quat.setRPY(0, 0, 0);
    }
    else
    {
        axis /= angle;
        tf2_quat.setRotation(axis, angle);
    }

    return tf2::toMsg(tf2_quat);
}
