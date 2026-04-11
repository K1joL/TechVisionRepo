#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

namespace lib {
using nlohmann::json;

using BoundingBox = std::vector<cv::Point2f>;
using Corners = std::vector<cv::Point2f>;
using ObjPoints = std::vector<cv::Point3f>;
using ImgPoints = std::vector<cv::Point3f>;

std::string decodeQrCode(cv::Mat& image, BoundingBox& bbox);
std::string decodeQrCode(cv::Mat& image);

void putTextCentered(cv::Mat& image, const std::string& text, cv::Point center,
                     int fontFace = cv::FONT_HERSHEY_SIMPLEX,
                     double fontScale = 0.6,
                     cv::Scalar color = cv::Scalar(0, 255, 0),
                     int thickness = 1);

cv::Mat getTransformedQrCode(const cv::Mat& image, const Corners& corners);
float getQrAngle(const Corners& corners);
}  // namespace lib