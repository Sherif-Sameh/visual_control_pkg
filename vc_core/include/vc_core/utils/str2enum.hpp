/*
 * Description:
 * Utility unordered maps for mapping between string to enum entries.
 */

#ifndef STRING_TO_ENUM
#define STRING_TO_ENUM

#include <string_view>
#include <unordered_map>

#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/calib3d.hpp>

namespace utils
{
    namespace str2enum
    {
        inline const std::unordered_map<std::string_view, cv::aruco::PREDEFINED_DICTIONARY_NAME>
            cvArucoPredefinedDictionaryNameMap = {
                {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
                {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
                {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
                {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
                {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
                {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
                {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
                {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
                {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
                {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
                {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
                {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
                {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
                {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
                {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
                {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
                {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
                {"DICT_APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
                {"DICT_APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
                {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
                {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11}};

        inline const std::unordered_map<std::string_view, cv::HandEyeCalibrationMethod>
            cvHandEyeCalibrationMethodMap = {
                {"CALIB_HAND_EYE_TSAI", cv::CALIB_HAND_EYE_TSAI},
                {"CALIB_HAND_EYE_PARK", cv::CALIB_HAND_EYE_PARK},
                {"CALIB_HAND_EYE_HORAUD", cv::CALIB_HAND_EYE_HORAUD},
                {"CALIB_HAND_EYE_ANDREFF", cv::CALIB_HAND_EYE_ANDREFF},
                {"CALIB_HAND_EYE_DANIILIDIS", cv::CALIB_HAND_EYE_DANIILIDIS}};
    } // namespace str2enum
} // namespace utils

#endif