/*
 * Description:
 * Utility functions for converting between different geometrical representations.
 */

#ifndef VC_UTILS_GEOMETRY
#define VC_UTILS_GEOMETRY

#include <cassert>
#include <vector>

#include <Eigen/Core>
#include <manif/SE3.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpQuaternionVector.h>
#include <visp3/core/vpTranslationVector.h>

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace utils
{
    namespace geometry
    {
        /**
         * @brief Convert pose from `geometry_msgs::msg::Pose` to ViSP `vpHomogeneousMatrix`.
         *
         * @param[in] pose Input pose to initialize `vpHomogeneousMatrix` from.
         * @return ViSP `vpHomogeneousMatrix` initialized from input pose.
         */
        inline vpHomogeneousMatrix gm_pose_to_vp_hmatrix(const geometry_msgs::msg::Pose &pose)
        {
            return vpHomogeneousMatrix(
                vpTranslationVector(pose.position.x, pose.position.y, pose.position.z),
                vpQuaternionVector(pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                   pose.orientation.w));
        }

        /**
         * @brief Convert pose from `geometry_msgs::msg::Transform` to ViSP `vpHomogeneousMatrix`.
         *
         * @param[in] transform Input transform to initialize `vpHomogeneousMatrix` from.
         * @return ViSP `vpHomogeneousMatrix` initialized from input pose
         */
        inline vpHomogeneousMatrix
        gm_transform_to_vp_hmatrix(const geometry_msgs::msg::Transform &transform)
        {
            return vpHomogeneousMatrix(
                vpTranslationVector(transform.translation.x, transform.translation.y,
                                    transform.translation.z),
                vpQuaternionVector(transform.rotation.x, transform.rotation.y, transform.rotation.z,
                                   transform.rotation.w));
        }

        template <typename Scalar>
        vpHomogeneousMatrix mnf_se3_to_vp_hmatrix(const manif::SE3<Scalar> &x)
        {
            Eigen::Matrix<Scalar, 7, 1> x_coeffs = x.coeffs();
            return vpHomogeneousMatrix(
                vpTranslationVector(x_coeffs(0), x_coeffs(1), x_coeffs(2)),
                vpQuaternionVector(x_coeffs(3), x_coeffs(4), x_coeffs(5), x_coeffs(6)));
        }

        template <typename Scalar, bool normalize = false>
        manif::SE3<Scalar> gm_transform_to_mnf_se3(const geometry_msgs::msg::Transform &transform)
        {
            Eigen::Matrix<Scalar, 3, 1> t{transform.translation.x, transform.translation.y,
                                          transform.translation.z};
            Eigen::Quaternion<Scalar> q(transform.rotation.w, transform.rotation.x,
                                        transform.rotation.y, transform.rotation.z);
            if constexpr (normalize)
            {
                q.normalize();
            }
            return manif::SE3<Scalar>(t, q);
        }

        template <typename Scalar>
        manif::SE3<Scalar> vp_hmatrix_to_mnf_se3(const vpHomogeneousMatrix &hm)
        {
            std::vector<Scalar> hm_coeffs;
            hm.convert(hm_coeffs);
            return manif::SE3<Scalar>(
                Eigen::Matrix<Scalar, 3, 1>{hm_coeffs[3], hm_coeffs[7], hm_coeffs[11]},
                Eigen::Quaternion<Scalar>(Eigen::Matrix<Scalar, 3, 3>{
                    hm_coeffs[0], hm_coeffs[1], hm_coeffs[2], hm_coeffs[4], hm_coeffs[5],
                    hm_coeffs[6], hm_coeffs[8], hm_coeffs[9], hm_coeffs[10]}));
        }

        /**
         * @brief Convert rotation from (x, y, z) 3D axis-angle to
         * `geometry_msgs::msg::Quaternion`.
         *
         * @param[in] x x-component of the axis-angle representation.
         * @param[in] y y-component of the axis-angle representation.
         * @param[in] z z-component of the axis-angle representation.
         * @return `geometry_msgs::msg::Quaternion` initialized from input axis-angle.
         */
        inline geometry_msgs::msg::Quaternion xyz_aa_to_gm_quat(const double x, const double y,
                                                                const double z)
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

        /**
         * @brief Convert a `geometry_msgs::msg::Pose` into a separate Eigen translation vector and
         * quaternion.
         *
         * @tparam Scalar Scalar type of the output Eigen arguments.
         * @tparam normalize Normalize quaternion after conversion.
         * @param[in] pose Input `geometry_msgs::msg::Pose` to read data from.
         * @param[out] t Output translation vector.
         * @param[out] q Output quaternion rotation.
         */
        template <typename Scalar, bool normalize = false>
        void gm_pose_to_eigen_tq(const geometry_msgs::msg::Pose &pose,
                                 Eigen::Matrix<Scalar, 3, 1> &t, Eigen::Quaternion<Scalar> &q)
        {
            t << static_cast<Scalar>(pose.position.x), static_cast<Scalar>(pose.position.y),
                static_cast<Scalar>(pose.position.z);
            Eigen::Quaternion<Scalar> q_tmp(
                static_cast<Scalar>(pose.orientation.w), static_cast<Scalar>(pose.orientation.x),
                static_cast<Scalar>(pose.orientation.y), static_cast<Scalar>(pose.orientation.z));
            if constexpr (normalize)
            {
                q_tmp.normalize();
            }
            q = std::move(q_tmp);
        }
    } // namespace geometry
} // namespace utils

#endif