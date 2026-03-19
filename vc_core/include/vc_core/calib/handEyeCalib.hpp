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

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <yaml-cpp/yaml.h>

namespace calib
{
    template <typename Scalar>
    class HandEyeCalib
    {
    public:
        using Isometry3x = Eigen::Transform<Scalar, 3, Eigen::Isometry>;

    public:
        HandEyeCalib();
        HandEyeCalib(const bool verbose, const std::size_t n_poses, const Scalar dist_to_target,
                     const cv::HandEyeCalibrationMethod calib_method);

        void setVerbose(const bool verbose);
        void setNPoses(const std::size_t n_poses, const bool resample = true);
        void setDistToTarget(const Scalar dist_to_target, const bool resample = true);
        void setCalibMethod(const cv::HandEyeCalibrationMethod calib_method);

        void addPoses(const Isometry3x &base_ee, const Isometry3x &cam_target);
        bool getNextTargetPose(Isometry3x &cam_target) const;
        bool calibrateHandEye(const std::string &path) const;

    protected:
        void sampleTargetPoses();
        void sampleRotations(std::vector<Eigen::AngleAxis<Scalar>> &rotations) const;
        bool writeToYaml(const std::string &path, const Isometry3x &ee_cam) const;

    protected:
        bool m_verbose;
        std::size_t m_n_poses;
        Scalar m_dist_to_target;
        cv::HandEyeCalibrationMethod m_calib_method;
        std::vector<Isometry3x> m_base_ee;
        std::vector<Isometry3x> m_cam_target;
    };

    // Definitions
    template <typename Scalar>
    HandEyeCalib<Scalar>::HandEyeCalib()
        : m_verbose(false), m_n_poses(0), m_dist_to_target(0),
          m_calib_method(cv::HandEyeCalibrationMethod::CALIB_HAND_EYE_TSAI)
    {
    }

    template <typename Scalar>
    HandEyeCalib<Scalar>::HandEyeCalib(const bool verbose, const std::size_t n_poses,
                                       const Scalar dist_to_target,
                                       const cv::HandEyeCalibrationMethod calib_method)
    {
        setVerbose(verbose);
        setNPoses(n_poses, false);
        setDistToTarget(dist_to_target, true);
        setCalibMethod(calib_method);
    }

    template <typename Scalar>
    void HandEyeCalib<Scalar>::setVerbose(const bool verbose)
    {
        if (verbose)
        {
            std::cout << "HandEyeCalib Settings:\n";
            std::cout << "verbose: " << verbose << "\n" << std::endl;
        }
        m_verbose = verbose;
    }
    template <typename Scalar>
    void HandEyeCalib<Scalar>::setNPoses(const std::size_t n_poses, const bool resample)
    {
        assert(n_poses > 2);
        if (m_verbose)
        {
            std::cout << "HandEyeCalib Settings:\n";
            std::cout << "n_poses: " << n_poses << "\n" << std::endl;
        }
        m_n_poses = n_poses;
        if (resample) sampleTargetPoses();
    }

    template <typename Scalar>
    void HandEyeCalib<Scalar>::setDistToTarget(const Scalar dist_to_target, const bool resample)
    {
        assert(dist_to_target > 0);
        if (m_verbose)
        {
            std::cout << "HandEyeCalib Settings:\n";
            std::cout << "dist_to_target: " << dist_to_target << "\n" << std::endl;
        }
        m_dist_to_target = dist_to_target;
        if (resample) sampleTargetPoses();
    }

    template <typename Scalar>
    void HandEyeCalib<Scalar>::setCalibMethod(const cv::HandEyeCalibrationMethod calib_method)
    {
        if (m_verbose)
        {
            std::cout << "HandEyeCalib Settings:\n";
            std::cout << "calib_method: " << calib_method << "\n" << std::endl;
        }
        m_calib_method = calib_method;
    }

    template <typename Scalar>
    void HandEyeCalib<Scalar>::addPoses(const Isometry3x &base_ee, const Isometry3x &cam_target)
    {
        std::size_t idx = m_base_ee.size();
        if (idx == m_n_poses)
        {
            if (m_verbose)
            {
                std::cout << "HandEyeCalib Information:\n";
                std::cout << "Pose buffers are full. Ignoring input poses.\n" << std::endl;
            }
            return;
        }
        m_base_ee.push_back(base_ee);
        m_cam_target[idx] = cam_target;
        if (m_verbose)
        {
            std::cout << "HandEyeCalib Information:\n";
            std::cout << "Gathered" << m_base_ee.size() << " poses so far.\n" << std::endl;
        }
    }

    template <typename Scalar>
    bool HandEyeCalib<Scalar>::getNextTargetPose(Isometry3x &cam_target) const
    {
        std::size_t idx = m_base_ee.size();
        if (idx == m_n_poses)
        {
            if (m_verbose)
            {
                std::cout << "HandEyeCalib Information:\n";
                std::cout << "Pose buffers are full. No new target poses to return.\n" << std::endl;
            }
            return false;
        }
        cam_target = m_cam_target[idx];
        return true;
    }

    template <typename Scalar>
    bool HandEyeCalib<Scalar>::calibrateHandEye(const std::string &path) const
    {
        assert(path.length() > 0);
        assert(m_base_ee.size() == m_n_poses);
        std::vector<cv::Mat, m_n_poses> R_base_ee, t_base_ee;
        std::vector<cv::Mat, m_n_poses> R_cam_target, t_cam_target;
        for (std::size_t i = 0; i < m_n_poses; i++)
        {
            cv::eigen2cv(m_base_ee[i].rotation(), R_base_ee[i]);
            cv::eigen2cv(m_base_ee[i].translation(), t_base_ee[i]);
            cv::eigen2cv(m_cam_target[i].rotation(), R_cam_target[i]);
            cv::eigen2cv(m_cam_target[i].translation(), t_cam_target[i]);
        }

        cv::Mat R_ee_cam(3, 3, cv::DataType<Scalar>::type);
        cv::Mat t_ee_cam(3, 1, cv::DataType<Scalar>::type);
        cv::calibrateHandEye(R_base_ee, t_base_ee, R_cam_target, t_cam_target, R_ee_cam, t_ee_cam,
                             m_calib_method);

        Eigen::Matrix<Scalar, 3, 3> R_ee_cam_eigen;
        Eigen::Matrix<Scalar, 3, 1> t_ee_cam_eigen;
        cv::cv2eigen(R_ee_cam, R_ee_cam_eigen);
        cv::cv2eigen(t_ee_cam, t_ee_cam_eigen);
        Isometry3x ee_cam = Isometry3x::Identity();
        ee_cam.translation() = t_ee_cam_eigen;
        ee_cam.rotation() = R_ee_cam_eigen;

        bool write_success = writeToYaml(path, ee_cam);
        if (m_verbose && write_success)
        {
            std::cout << "HandEyeCalib Information:\n";
            std::cout << "Calibration output written to " << path << "\n" << std::endl;
        }
        return write_success;
    }

    template <typename Scalar>
    void HandEyeCalib<Scalar>::sampleTargetPoses()
    {
        assert(m_n_poses > 2);
        assert(m_dist_to_target > 0);
        m_base_ee.clear();
        m_cam_target.resize(m_n_poses);

        Isometry3x target_cam_0 = Isometry3x::Identity();
        target_cam_0.translate(Eigen::Matrix<Scalar, 3, 1>(0, 0, -m_dist_to_target));

        std::vector<Eigen::AngleAxis<Scalar>> rotations;
        sampleRotations(rotations);
        for (std::size_t i = 0; i < m_n_poses; i++)
        {
            Isometry3x target_cam_i = Isometry3x(target_cam_0).prerotate(rotations[i]);
            m_cam_target[i] = target_cam_i.inverse();
        }

        if (m_verbose)
        {
            std::cout << "HandEyeCalib Information:\n";
            std::cout << "Sampled " << m_n_poses << " new target poses.\n" << std::endl;
        }
    }

    template <typename Scalar>
    void
    HandEyeCalib<Scalar>::sampleRotations(std::vector<Eigen::AngleAxis<Scalar>> &rotations) const
    {
        Scalar angle = static_cast<Scalar>(M_PI) / m_n_poses;
        rotations.resize(m_n_poses);
        for (std::size_t i = 0; i < m_n_poses; i++)
        {
            Eigen::Matrix<Scalar, 3, 1> axis(std::cos(i * angle), std::sin(i * angle), 0);
            rotations[i] = Eigen::AngleAxis<Scalar>(angle, axis);
        }
    }

    template <typename Scalar>
    bool HandEyeCalib<Scalar>::writeToYaml(const std::string &path, const Isometry3x &ee_cam) const
    {
        YAML::Emitter out;
        out << YAML::BeginMap;
        // write out calibration config info
        out << YAML::Key << "config" << YAML::Value;
        out << YAML::BeginMap;
        out << YAML::Key << "n_poses" << YAML::Value << m_n_poses;
        out << YAML::Key << "dist_to_target" << YAML::Value << m_dist_to_target;
        out << YAML::Key << "calib_method" << YAML::Value << m_calib_method;
        out << YAML::EndMap;

        // write out calibration output pose
        auto t = ee_cam.translation();
        Eigen::Quaternion<Scalar> q(ee_cam.rotation());
        out << YAML::Key << "pose" << YAML::Value;
        out << YAML::BeginMap;
        out << YAML::Key << "translation" << YAML::Value << YAML::Flow;
        out << YAML::BeginSeq << t.x() << t.y() << t.z() << YAML::EndSeq;
        out << YAML::Key << "rotation" << YAML::Value << YAML::Flow;
        out << YAML::BeginSeq << q.w() << q.x() << q.y() << q.z() << YAML::EndSeq;
        out << YAML::EndMap << YAML::EndMap;

        // write out to yaml file
        std::ofstream file(path);
        if (m_verbose && !file)
        {
            std::cerr << "HandEyeCalib Error:\n";
            std::cerr << "Could not open file at " << path < < < < "\n" << std::endl;
            return false;
        }
        file << out.c_str();
        if (file.bad())
        {
            std::cerr << "HandEyeCalib Error:\n";
            std::cerr << "Could not write to file at " << path < < < < "\n" << std::endl;
            return false;
        }
        file.close();
        return true;
    }
} // namespace calib
#endif