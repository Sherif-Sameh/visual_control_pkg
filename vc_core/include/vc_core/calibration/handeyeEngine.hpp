/*
 * Description:
 * Implements hand-eye calibration procedure using OpenCV.
 */

#ifndef CALIBRATION_HANDEYE
#define CALIBRATION_HANDEYE

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <toml++/toml.hpp>

namespace calib
{
    /**
     * @brief Generic template for performing hand-eye calibration.
     *
     * This class template implements the following four main functionalities:
     *
     * 1) Target pose sampling according to the method described in the paper by Tsai and Lenz
     * titled `A new technique for fully autonomous and efficient 3D robotics hand-eye calibration`.
     *
     * 2) Recording of the corresponding EE wrt base and target wrt camera `n_poses` needed for
     * calibration.
     *
     * 3) Retreiving of the next camera wrt target pose that should be reached within the sampled
     * target `n_poses`.
     *
     * 4) Solving for the calibration parameters (i.e. pose of the camera wrt the EE) and writing
     * them into a `toml` config file.
     *
     * Internally, `OpenCV` is used for solving for the hand-eye calibration parameters
     * (i.e. the pose of the camera wrt the end-effector). Externally, an `Eigen`-based interface is
     * used instead.
     * @tparam Scalar Scalar type to use for `Eigen` and `OpenCV` objects.
     */
    template <typename Scalar>
    class HandeyeEngine
    {
    public:
        using Isometry3x = Eigen::Transform<Scalar, 3, Eigen::Isometry>;

    public:
        HandeyeEngine();
        HandeyeEngine(const std::size_t n_poses, const Scalar rot_angle,
                      const Scalar dist_to_target, const cv::HandEyeCalibrationMethod calib_method);

        std::size_t getNPosesStored() const;
        /**
         * @brief Set the number of target poses and optionally resample them.
         *
         * @param[in] n_poses Number of target poses. Must be > 2.
         * @param[in] resample Resample target poses after update. Defaults to `true`.
         */
        void setNPoses(const std::size_t n_poses, const bool resample = true);
        /**
         * @brief Set the rotation angle for sampling target poses and optionally resample them.
         *
         * @param[in] rot_angle Rotation angle in radians for generating target pose samples.
         * @param[in] resample Resample target poses after update. Defaults to `true`.
         */
        void setRotAngle(const Scalar rot_angle, const bool resample = true);
        /**
         * @brief Set the distance of the camera to the target for target poses and optionally
         * resample them.
         *
         * @param[in] dist_to_target Distance of the camera to the target. Must be > 0.
         * @param[in] resample Resample target poses after update. Defaults to `true`.
         */
        void setDistToTarget(const Scalar dist_to_target, const bool resample = true);
        void setCalibMethod(const cv::HandEyeCalibrationMethod calib_method);

        /**
         * @brief Add the new corresponding EE wrt base and target wrt camera poses to the recorded
         * poses.
         *
         * If the vectors for recorded poses are already full, the inputs are ignored.
         *
         * @param[in] base_ee Pose of the EE wrt the base.
         * @param[in] cam_target Pose of the target wrt the camera.
         */
        void addPoses(const Isometry3x &base_ee, const Isometry3x &cam_target);
        /**
         * @brief Get the next camera wrt target pose that should be reached within the sampled
         * target poses.
         *
         * @param[out] target_cam Next camera wrt target pose to track. Is not updated if all target
         * poses have already been processed.
         * @return `true` if there are remaining target poses and `target_cam` has been updated
         * correctly.
         * @return `false` otherwise.
         */
        bool getNextCameraPose(Isometry3x &target_cam) const;
        /**
         * @brief Solve for the calibration parameters and write them out to a `toml` config file.
         *
         * @param[in] path Path for the config `.toml` file for writing calibration parameters. If
         * empty, no write attempts are made.
         * @param[out] ee_cam Computed pose of camera wrt EE from calibration procedure.
         * @return `true` if calibration parameters were written successfully to the config file.
         * @return `false` otherwise.
         */
        bool calibrateHandEye(const std::string &path, Isometry3x &ee_cam) const;

    protected:
        void sampleTargetPoses();
        void sampleRotations(std::vector<Eigen::AngleAxis<Scalar>> &rotations) const;
        bool writeToFile(const std::string &path, const Isometry3x &ee_cam) const;

    protected:
        std::size_t m_n_poses;
        Scalar m_rot_angle;
        Scalar m_dist_to_target;
        cv::HandEyeCalibrationMethod m_calib_method;
        std::vector<Isometry3x> m_target_cam;
        std::vector<Isometry3x> m_base_ee;
        std::vector<Isometry3x> m_cam_target;
    };

    // Definitions
    template <typename Scalar>
    HandeyeEngine<Scalar>::HandeyeEngine()
        : m_n_poses(0), m_dist_to_target(0), m_calib_method(cv::CALIB_HAND_EYE_TSAI)
    {
    }

    template <typename Scalar>
    HandeyeEngine<Scalar>::HandeyeEngine(const std::size_t n_poses, const Scalar rot_angle,
                                         const Scalar dist_to_target,
                                         const cv::HandEyeCalibrationMethod calib_method)
    {
        setNPoses(n_poses, false);
        setRotAngle(rot_angle, false);
        setDistToTarget(dist_to_target, true);
        setCalibMethod(calib_method);
    }

    template <typename Scalar>
    std::size_t HandeyeEngine<Scalar>::getNPosesStored() const
    {
        return m_base_ee.size();
    }

    template <typename Scalar>
    void HandeyeEngine<Scalar>::setNPoses(const std::size_t n_poses, const bool resample)
    {
        assert(n_poses > 2);
        m_n_poses = n_poses;
        if (resample) sampleTargetPoses();
    }

    template <typename Scalar>
    void HandeyeEngine<Scalar>::setRotAngle(const Scalar rot_angle, const bool resample)
    {
        assert(rot_angle > 0);
        m_rot_angle = rot_angle;
        if (resample) sampleTargetPoses();
    }

    template <typename Scalar>
    void HandeyeEngine<Scalar>::setDistToTarget(const Scalar dist_to_target, const bool resample)
    {
        assert(dist_to_target > 0);
        m_dist_to_target = dist_to_target;
        if (resample) sampleTargetPoses();
    }

    template <typename Scalar>
    void HandeyeEngine<Scalar>::setCalibMethod(const cv::HandEyeCalibrationMethod calib_method)
    {
        m_calib_method = calib_method;
    }

    template <typename Scalar>
    void HandeyeEngine<Scalar>::addPoses(const Isometry3x &base_ee, const Isometry3x &cam_target)
    {
        std::size_t idx = m_base_ee.size();
        if (idx == m_n_poses) return;

        m_base_ee.push_back(base_ee);
        m_cam_target.push_back(cam_target);
    }

    template <typename Scalar>
    bool HandeyeEngine<Scalar>::getNextCameraPose(Isometry3x &target_cam) const
    {
        std::size_t idx = m_base_ee.size();
        if (idx == m_n_poses) return false;

        target_cam = m_target_cam[idx];
        return true;
    }

    template <typename Scalar>
    bool HandeyeEngine<Scalar>::calibrateHandEye(const std::string &path, Isometry3x &ee_cam) const
    {
        assert(m_base_ee.size() == m_n_poses);
        std::vector<cv::Mat> R_base_ee(m_n_poses), t_base_ee(m_n_poses);
        std::vector<cv::Mat> R_cam_target(m_n_poses), t_cam_target(m_n_poses);
        for (std::size_t i = 0; i < m_n_poses; i++)
        {
            cv::eigen2cv(m_base_ee[i].rotation().eval(), R_base_ee[i]);
            cv::eigen2cv(m_base_ee[i].translation().eval(), t_base_ee[i]);
            cv::eigen2cv(m_cam_target[i].rotation().eval(), R_cam_target[i]);
            cv::eigen2cv(m_cam_target[i].translation().eval(), t_cam_target[i]);
        }

        cv::Mat R_ee_cam(3, 3, cv::DataType<Scalar>::type);
        cv::Mat t_ee_cam(3, 1, cv::DataType<Scalar>::type);
        cv::calibrateHandEye(R_base_ee, t_base_ee, R_cam_target, t_cam_target, R_ee_cam, t_ee_cam,
                             m_calib_method);

        Eigen::Matrix<Scalar, 3, 3> R_ee_cam_eigen;
        Eigen::Matrix<Scalar, 3, 1> t_ee_cam_eigen;
        cv::cv2eigen(R_ee_cam, R_ee_cam_eigen);
        cv::cv2eigen(t_ee_cam, t_ee_cam_eigen);
        ee_cam = Isometry3x::Identity();
        ee_cam.translation() = t_ee_cam_eigen;
        ee_cam.linear() = R_ee_cam_eigen;

        if (path.length() == 0) return true;
        bool write_success = writeToFile(path, ee_cam);
        if (write_success)
        {
            std::cout << "Calibration output written to " << path << "\n" << std::endl;
        }
        return write_success;
    }

    template <typename Scalar>
    void HandeyeEngine<Scalar>::sampleTargetPoses()
    {
        assert(m_n_poses > 2);
        assert(m_dist_to_target > 0);
        m_base_ee.clear();
        m_cam_target.clear();
        m_target_cam.resize(m_n_poses);

        Isometry3x target_cam_0 = Isometry3x::Identity();
        target_cam_0.translate(Eigen::Matrix<Scalar, 3, 1>(0, 0, -m_dist_to_target));

        std::vector<Eigen::AngleAxis<Scalar>> rotations;
        sampleRotations(rotations);
        for (std::size_t i = 0; i < m_n_poses; i++)
        {
            m_target_cam[i] = Isometry3x(target_cam_0).prerotate(rotations[i]);
        }
    }

    template <typename Scalar>
    void
    HandeyeEngine<Scalar>::sampleRotations(std::vector<Eigen::AngleAxis<Scalar>> &rotations) const
    {
        Scalar angle = static_cast<Scalar>(2 * M_PI) / m_n_poses;
        rotations.resize(m_n_poses);
        for (std::size_t i = 0; i < m_n_poses; i++)
        {
            Eigen::Matrix<Scalar, 3, 1> axis(std::cos(i * angle), std::sin(i * angle), 0);
            rotations[i] = Eigen::AngleAxis<Scalar>(m_rot_angle, axis);
        }
    }

    template <typename Scalar>
    bool HandeyeEngine<Scalar>::writeToFile(const std::string &path, const Isometry3x &ee_cam) const
    {
        auto t = ee_cam.translation();
        Eigen::Quaternion<Scalar> q(ee_cam.rotation());
        auto tbl = toml::table{
            {"config", toml::table{{"n_poses", static_cast<int>(m_n_poses)},
                                   {"rot_angle", m_rot_angle},
                                   {"dist_to_target", m_dist_to_target},
                                   {"calib_method", m_calib_method}}},
            {"pose", toml::table{{"translation", toml::array{t.x(), t.y(), t.z()}},
                                 {"rotation", toml::array{q.w(), q.x(), q.y(), q.z()}}}},
        };

        // write to toml file
        std::ofstream file(path);
        if (!file)
        {
            std::cerr << "Could not open file at " << path << "\n" << std::endl;
            return false;
        }
        file << tbl;
        if (file.bad())
        {
            std::cerr << "Could not write to file at " << path << "\n" << std::endl;
            return false;
        }
        file.close();
        return true;
    }
} // namespace calib
#endif