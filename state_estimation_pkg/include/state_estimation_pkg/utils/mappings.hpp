/*
 * Description:
 * Utility functions for mapping data between different equivalent memory format.
 */

#ifndef SE_UTILS_MAPPINGS
#define SE_UTILS_MAPPINGS

#include <array>

#include <Eigen/Core>

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
    } // namespace mappings
} // namespace utils

#endif