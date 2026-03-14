/*
 * Description:
 * Utility functions for mapping data between different equivalent memory formats.
 */

#ifndef VC_UTILS_MAPPINGS
#define VC_UTILS_MAPPINGS

#include <array>
#include <vector>

#include <Eigen/Core>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <visp3/core/vpColVector.h>

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"

namespace utils
{
    namespace mappings
    {
        /**
         * @brief Converts a flattened vector data representation into an `Eigen::Matrix`.
         *
         * @tparam T Float type of the vector and matrix.
         * @tparam A Storage alignment of the data in the vector.
         * @param[in] vec Flattened vector of size = `n_rows` * `n_cols` to read data from.
         * @param[in] n_rows Number of row for the matrix.
         * @param[in] n_cols Number of columns for the matrix.
         * @param[out] mat Output matrix of type `Eigen::Matrix<T, n_rows, n_cols, align>`.
         */
        template <typename T, Eigen::StorageOptions A>
        void to_eigen_matrix(std::vector<T> &vec, const std::size_t n_rows,
                             const std::size_t n_cols,
                             Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
        {
            assert(vec.size() == (n_rows * n_cols));
            mat = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, A>>(
                vec.data(), static_cast<int>(n_rows), static_cast<int>(n_cols));
        }

        /**
         * @brief Converts a flattened vector data representation into a square `Eigen::Matrix`.
         *
         * @tparam T Float type of the vector and matrix.
         * @tparam A Storage alignment of the data in the vector.
         * @param[in] vec Flattened vector of size = `n_rows` * `n_cols` to read data from.
         * @param[out] mat Output matrix of type `Eigen::Matrix<T, n_rows, n_cols, align>`.
         */
        template <typename T, Eigen::StorageOptions A>
        void to_eigen_matrix(std::vector<T> &vec,
                             Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
        {
            std::size_t n_rows =
                static_cast<std::size_t>(std::sqrt(static_cast<double>(vec.size())));
            assert(vec.size() == (n_rows * n_rows));
            return to_eigen_matrix<T, A>(vec, n_rows, n_rows, mat);
        }

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
        to_array(const Eigen::MatrixBase<Derived> &mat)
        {
            using Scalar = typename Derived::Scalar;
            constexpr int Rows = Derived::RowsAtCompileTime;
            constexpr int Cols = Derived::ColsAtCompileTime;

            std::array<Scalar, Rows * Cols> arr;
            Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols, Align>>(arr.data()) = mat;
            return arr;
        }

        /**
         * @brief Convert a vector to a `KDL::JntArray`.
         *
         * @param[in] vec Vector to read data from.
         * @param[out] jntarray Output `KDL::JntArray` of the same size as the input vector.
         */
        inline void to_kdl_jntarray(const std::vector<double> &vec, KDL::JntArray &jntarray)
        {
            assert(vec.size() == static_cast<std::size_t>(jntarray.data.size()));
            for (std::size_t i = 0; i < vec.size(); i++)
            {
                jntarray(i) = vec[i];
            }
        }

        /**
         * @brief Convert a ViSP `vpColVector` to a `KDL::JntArray`.
         *
         * @param[in] vpcolvector `vpColVector` to read data from.
         * @param[out] jntarray Output `KDL::JntArray` of the same size as the input `vpColVector`.
         */
        inline void to_kdl_jntarray(const vpColVector &vpcolvector, KDL::JntArray &jntarray)
        {
            assert(vpcolvector.size() == static_cast<std::size_t>(jntarray.data.size()));
            for (std::size_t i = 0; i < vpcolvector.size(); i++)
            {
                jntarray(i) = vpcolvector[i];
            }
        }

        /**
         * @brief Convert a ViSP `vpColVector` to a `KDL::Twist`.
         *
         * @param[in] vpcolvector `vpColVector` to read data from whose size = 6.
         * @param[out] twist Output `KDL::Twist`.
         */
        inline void to_kdl_twist(const vpColVector &vpcolvector, KDL::Twist &twist)
        {
            assert(vpcolvector.size() == 6);
            for (std::size_t i = 0; i < 3; i++)
            {
                twist.vel.data[i] = vpcolvector[i];
                twist.rot.data[i] = vpcolvector[i + 3];
            }
        }

        /**
         * @brief Convert a `KDL::JntArray` to a ViSP `vpColVector`.
         *
         * @param[in] jntarray `KDL::JntArray` to read data from.
         * @param[out] vpcolvector Output `vpColVector` of the same size as the input
         * `KDL::JntArray`.
         */
        inline void to_visp_vpcolvector(const KDL::JntArray &jntarray, vpColVector &vpcolvector)
        {
            assert(vpcolvector.size() == static_cast<std::size_t>(jntarray.data.size()));
            for (std::size_t i = 0; i < vpcolvector.size(); i++)
            {
                vpcolvector[i] = jntarray(i);
            }
        }

        inline vpColVector to_vp_vpcolvector(const geometry_msgs::msg::Twist &twist)
        {
            return vpColVector({twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x,
                                twist.angular.y, twist.angular.z});
        }
    } // namespace mappings
} // namespace utils

#endif