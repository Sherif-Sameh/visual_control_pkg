/*
 * Description:
 * Utility functions for converting between different geometrical representations.
 */

#ifndef VC_UTILS_GEOMETRY
#define VC_UTILS_GEOMETRY

#include <cassert>
#include <vector>

#include <Eigen/Geometry>
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
        inline vpHomogeneousMatrix to_vp_hmatrix(const geometry_msgs::msg::Pose &pose)
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
         * @return ViSP `vpHomogeneousMatrix` initialized from input pose.
         */
        inline vpHomogeneousMatrix to_vp_hmatrix(const geometry_msgs::msg::Transform &transform)
        {
            return vpHomogeneousMatrix(
                vpTranslationVector(transform.translation.x, transform.translation.y,
                                    transform.translation.z),
                vpQuaternionVector(transform.rotation.x, transform.rotation.y, transform.rotation.z,
                                   transform.rotation.w));
        }

        /**
         * @brief Convert pose from `manif::SE3` to ViSP `vpHomogeneousMatrix`.
         *
         * @tparam Scalar Scalar type of the input `manif::SE3` Lie group.
         * @param[in] x Input SE3 pose to initialize `vpHomogeneousMatrix` from.
         * @return ViSP `vpHomogeneousMatrix` initialized from input pose.
         */
        template <typename Scalar>
        vpHomogeneousMatrix to_vp_hmatrix(const manif::SE3<Scalar> &x)
        {
            Eigen::Matrix<Scalar, 7, 1> x_coeffs = x.coeffs();
            return vpHomogeneousMatrix(
                vpTranslationVector(x_coeffs(0), x_coeffs(1), x_coeffs(2)),
                vpQuaternionVector(x_coeffs(3), x_coeffs(4), x_coeffs(5), x_coeffs(6)));
        }

        /**
         * @brief Convert pose from `geometry_msgs::msg::Pose` to `manif::SE3`.
         *
         * @tparam Scalar Scalar type of the output `manif::SE3` Lie group.
         * @tparam normalize Normalize quaternion after conversion. Defaults to false.
         * @param[in] pose Input pose to initialize SE3 pose from.
         * @return `manif::SE3<Scalar>` initialized from input pose.
         */
        template <typename Scalar, bool normalize = false>
        manif::SE3<Scalar> to_mnf_se3(const geometry_msgs::msg::Pose &pose)
        {
            Eigen::Matrix<Scalar, 3, 1> t{pose.position.x, pose.position.y, pose.position.z};
            Eigen::Quaternion<Scalar> q(pose.orientation.w, pose.orientation.x, pose.orientation.y,
                                        pose.orientation.z);
            if constexpr (normalize)
            {
                q.normalize();
            }
            return manif::SE3<Scalar>(t, q);
        }

        /**
         * @brief Convert pose from `geometry_msgs::msg::Transform` to `manif::SE3`.
         *
         * @tparam Scalar Scalar type of the output `manif::SE3` Lie group.
         * @tparam normalize Normalize quaternion after conversion. Defaults to false.
         * @param[in] transform Input transform to initialize SE3 pose from.
         * @return `manif::SE3<Scalar>` initialized from input pose.
         */
        template <typename Scalar, bool normalize = false>
        manif::SE3<Scalar> to_mnf_se3(const geometry_msgs::msg::Transform &transform)
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

        /**
         * @brief Convert pose from ViSP `vpHomogeneousMatrix` to `manif::SE3`.
         *
         * @tparam Scalar Scalar type of the output `manif::SE3` Lie group.
         * @tparam normalize Normalize quaternion after conversion. Defaults to false.
         * @param[in] hm Input homogeneous matrix to initialize SE3 pose from.
         * @return `manif::SE3<Scalar>` initialized from input pose.
         */
        template <typename Scalar, bool normalize = false>
        manif::SE3<Scalar> to_mnf_se3(const vpHomogeneousMatrix &hm)
        {
            std::vector<Scalar> hm_coeffs;
            hm.convert(hm_coeffs);
            Eigen::Matrix<Scalar, 3, 1> t{hm_coeffs[3], hm_coeffs[7], hm_coeffs[11]};
            Eigen::Quaternion<Scalar> q(Eigen::Matrix<Scalar, 3, 3>{
                hm_coeffs[0], hm_coeffs[1], hm_coeffs[2], hm_coeffs[4], hm_coeffs[5], hm_coeffs[6],
                hm_coeffs[8], hm_coeffs[9], hm_coeffs[10]});
            if constexpr (normalize)
            {
                q.normalize();
            }
            return manif::SE3<Scalar>(t, q);
        }

        /**
         * @brief Convert rotation from (x, y, z) 3D rotation vector to `Eigen::AngleAxis`.
         *
         * @tparam Scalar Scalar type of the inputs components and output `Eigen::AngleAxis`.
         * @param[in] x x-component of the rotation vector representation.
         * @param[in] y y-component of the rotation vector representation.
         * @param[in] z z-component of the rotation vector representation.
         * @return `Eigen::AngleAxis<Scalar>` initialized from input rotation vector.
         */
        template <typename Scalar>
        Eigen::AngleAxis<Scalar> to_eigen_angle_axis(const Scalar x, const Scalar y, const Scalar z)
        {
            constexpr double angle_threshold = 1e-6;
            Eigen::Matrix<Scalar, 3, 1> rvec(x, y, z);
            Scalar angle = rvec.norm();
            if (angle < angle_threshold)
            {
                return Eigen::AngleAxis<Scalar>(Eigen::Quaternion<Scalar>::Identity());
            }
            return Eigen::AngleAxis<Scalar>(angle, rvec.normalized());
        }

        /**
         * @brief Convert rotation from (x, y, z) 3D rotation vector to
         * `geometry_msgs::msg::Quaternion`.
         *
         * @param[in] x x-component of the rotation vector representation.
         * @param[in] y y-component of the rotation vector representation.
         * @param[in] z z-component of the rotation vector representation.
         * @return `geometry_msgs::msg::Quaternion` initialized from input rotation vector.
         */
        inline geometry_msgs::msg::Quaternion to_gm_quat(const double x, const double y,
                                                         const double z)
        {
            constexpr double angle_threshold = 1e-6;

            // Construct tf2 Quaternion from rotation vector representation
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
    } // namespace geometry
} // namespace utils

#endif