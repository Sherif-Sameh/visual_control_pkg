/*
 * Description:
 * Utility functions for mapping data between different equivalent memory format.
 */

#ifndef SE_UTILS_MAPPINGS
#define SE_UTILS_MAPPINGS

#include <array>

#include <Eigen/Core>

#include "geometry_msgs/msg/pose.hpp"

namespace utils
{
    namespace mappings
    {
        /**
         * @brief Converts a derived class from `Eigen::MatrixBase` into a flattened `std::array`.
         *
         * @tparam Derived Derived class from `Eigen::MatrixBase`.
         * @tparam Align Desired storage alignment of the data in output array.
         * @param[in] mat Input matrix to read data from.
         * @return `std::array<typename Derived::Scalar,
         * Derived::RowsAtCompileTime * Derived::ColsAtCompileTime>` Flattened array representation
         * of input matrix with the given storage alignment.
         */
        template <typename Derived, Eigen::StorageOptions Align>
        std::array<typename Derived::Scalar,
                   Derived::RowsAtCompileTime * Derived::ColsAtCompileTime>
        eigen_matrix_to_array(const Eigen::MatrixBase<Derived> &mat)
        {
            using Scalar = typename Derived::Scalar;
            constexpr int Rows = Derived::RowsAtCompileTime;
            constexpr int Cols = Derived::ColsAtCompileTime;

            std::array<Scalar, Rows * Cols> arr;
            Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols, Align>>(arr.data()) = mat;
            return arr;
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
        template <typename Scalar, bool normalize>
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
    } // namespace mappings
} // namespace utils

#endif