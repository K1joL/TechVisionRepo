#include "lib.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace lib {
std::string decodeQrCode(cv::Mat& image, BoundingBox& bbox) {
    cv::QRCodeDetector detector;
    std::string decodedData = detector.detectAndDecode(image, bbox);
    // if (!decodedData.empty() && !bbox.empty()) {
    std::cout << "Decoded data: " << decodedData << std::endl;
    // }
    return decodedData;
}

std::string decodeQrCode(cv::Mat& image) {
    BoundingBox bbox;
    return decodeQrCode(image, bbox);
}

void putTextCentered(cv::Mat& image, const std::string& text, cv::Point center,
                     int fontFace, double fontScale, cv::Scalar color,
                     int thickness) {
    int baseline = 0;
    cv::Size textSize =
        cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

    cv::Point origin(center.x - textSize.width / 2,
                     center.y + textSize.height / 2);

    cv::putText(image, text, origin, fontFace, fontScale, color, thickness);
}

cv::Mat getTransformedQrCode(const cv::Mat& image, const Corners& qrCorners) {
    // Measure actual side lengths from the detected corners
    float w = cv::norm(qrCorners[1] - qrCorners[0]);
    float h = cv::norm(qrCorners[3] - qrCorners[0]);
    float side = std::max(w, h);
    side = std::max(side, 200.f);
    std::vector<cv::Point2f> targetCorners = {
        {0, 0}, {side, 0}, {side, side}, {0, side}};
    cv::Mat transformTable =
        cv::getPerspectiveTransform(qrCorners, targetCorners);
    cv::Mat warped;
    cv::warpPerspective(image, warped, transformTable, {side, side},
                        cv::INTER_CUBIC);

    int pad = side / 5;
    cv::Mat padded;
    cv::copyMakeBorder(warped, padded, pad, pad, pad, pad, cv::BORDER_CONSTANT,
                       cv::Scalar(255, 255, 255));

    return padded;
}

float getQrAngle(const Corners& corners) {
    // corners order: TL, TR, BR, BL
    float topEdge    = cv::norm(corners[1] - corners[0]);
    float bottomEdge = cv::norm(corners[2] - corners[3]);

    // ratio = 1.0 → face-on, < 1.0 → top tilted away
    float ratio = topEdge / bottomEdge;
    return std::acos(std::min(ratio, 1.f)) * 180.f / CV_PI;
}
}  // namespace lib