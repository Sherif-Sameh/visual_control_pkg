/*
 * Description:
 * Utility functions for mapping data between different equivalent memory format.
 */

#ifndef UTILS_CONVERSIONS_MAPPINGS
#define UTILS_CONVERSIONS_MAPPINGS

#include <cassert>
#include <vector>

#include <Eigen/Core>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <visp3/core/vpColVector.h>

namespace mappings
{
    /**
     * @brief Converts a flattened vector data representation into an Eigen::Matrix.
     *
     * @tparam T Float type of the vector and matrix.
     * @tparam A Storage alignment of the data in the vector.
     * @param vec Flattened vector of size = `n_rows` * `n_cols` to read data from.
     * @param n_rows Number of row for the matrix.
     * @param n_cols Number of columns for the matrix.
     * @param mat Output matrix of type `Eigen::Matrix<T, n_rows, n_cols, align>`.
     */
    template <typename T, Eigen::StorageOptions A>
    void vec_to_eigen_matrix(std::vector<T> &vec, const std::size_t n_rows,
                             const std::size_t n_cols,
                             Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
    {
        assert(vec.size() == (n_rows * n_cols));
        mat = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, A>>(
            vec.data(), static_cast<int>(n_rows), static_cast<int>(n_cols));
    }

    /**
     * @brief Converts a flattened vector data representation into a square Eigen::Matrix.
     *
     * @tparam T Float type of the vector and matrix.
     * @tparam A Storage alignment of the data in the vector.
     * @param vec Flattened vector of size = `n_rows` * `n_cols` to read data from.
     * @param mat Output matrix of type `Eigen::Matrix<T, n_rows, n_cols, align>`.
     */
    template <typename T, Eigen::StorageOptions A>
    void vec_to_sqr_eigen_matrix(std::vector<T> &vec,
                                 Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat)
    {
        std::size_t n_rows = static_cast<std::size_t>(std::sqrt(static_cast<double>(vec.size())));
        assert(vec.size() == (n_rows * n_rows));
        return vec_to_eigen_matrix<T, A>(vec, n_rows, n_rows, mat);
    }

    /**
     * @brief Convert a vector to a KDL::JntArray
     *
     * @param vec Vector to read data from.
     * @param jntarray Output KDL::JntArray of the same size as the input vector.
     */
    inline void vec_to_kdl_jntarray(const std::vector<double> &vec, KDL::JntArray &jntarray)
    {
        assert(vec.size() == static_cast<std::size_t>(jntarray.data.size()));
        for (std::size_t i = 0; i < vec.size(); i++)
        {
            jntarray(i) = vec[i];
        }
    }

    /**
     * @brief Convert a visp vpColVector to a KDL::JntArray
     *
     * @param vpcolvector vpColVector to read data from.
     * @param jntarray Output KDL::JntArray of the same size as the input vpColVector.
     */
    inline void visp_vpcolvector_to_kdl_jntarray(const vpColVector &vpcolvector,
                                                 KDL::JntArray &jntarray)
    {
        assert(vpcolvector.size() == static_cast<std::size_t>(jntarray.data.size()));
        for (std::size_t i = 0; i < vpcolvector.size(); i++)
        {
            jntarray(i) = vpcolvector[i];
        }
    }

    /**
     * @brief Convert a visp vpColVector to a KDL::Twist
     *
     * @param vpcolvector vpColVector to read data from whose size = 6.
     * @param twist Output KDL::Twist.
     */
    inline void visp_vpcolvector_to_kdl_twist(const vpColVector &vpcolvector, KDL::Twist &twist)
    {
        assert(vpcolvector.size() == 6);
        for (std::size_t i = 0; i < 3; i++)
        {
            twist.vel.data[i] = vpcolvector[i];
            twist.rot.data[i] = vpcolvector[i + 3];
        }
    }

    /**
     * @brief Convert a KDL::JntArray to a visp vpColVector
     *
     * @param jntarray KDL::JntArray to read data from.
     * @param vpcolvector Output vpColVector of the same size as the input KDL::JntArray.
     */
    inline void kdl_jntarray_to_visp_vpcolvector(const KDL::JntArray &jntarray,
                                                 vpColVector &vpcolvector)
    {
        assert(vpcolvector.size() == static_cast<std::size_t>(jntarray.data.size()));
        for (std::size_t i = 0; i < vpcolvector.size(); i++)
        {
            vpcolvector[i] = jntarray(i);
        }
    }
} // namespace mappings

#endif