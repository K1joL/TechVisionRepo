#include <getopt.h>

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "lib.hpp"

using cv::Mat, cv::VideoCapture;

int main(int argc, char* argv[]) {
    int opt;
    int cameraDev = 0;
    const char* options = "c:";

    while ((opt = getopt(argc, argv, options)) != -1) {
        switch (opt) {
            case 'c':
                cameraDev = std::stoi(optarg);
                break;
            default:
                std::cerr << "Error during option parsing" << std::endl;
                return EXIT_FAILURE;
        }
    }

    VideoCapture cap(cameraDev);
    if (!cap.isOpened()) {
        printf("Error: Cannot open camera\n");
        return -1;
    }
    Mat frame;

    cv::namedWindow("Webcam");
    std::string decodedData;
    lib::BoundingBox bbox;
    cv::QRCodeDetector detector;
    while (true) {
        cap >> frame;
        cv::Mat copyFrame;
        frame.copyTo(copyFrame);

        // Step 1: detect corners on the raw (possibly distorted) frame
        bool found = detector.detect(frame, bbox);
        if (found) {
            std::cout << "QR Code angle: " << lib::getQrAngle(bbox) << std::endl;
            cv::Mat warped = lib::getTransformedQrCode(frame, bbox);
            std::string data = lib::decodeQrCode(frame, bbox);
            for (int i = 0; i < 4; i++)
                cv::line(frame, bbox[i], bbox[(i + 1) % 4], {0, 255, 0}, 2);

            if (!data.empty())
                lib::putTextCentered(frame, data,
                    cv::Point2f((bbox[0] + bbox[2]) * 0.5f));

            cv::imshow("QRCode", warped);
        }
        // decodedData = lib::decodeQrCode(frame, bbox);
        // if (!decodedData.empty() && !bbox.empty()) {
        //     cv::Mat qrcode = lib::getTransformedQrCode(frame, bbox);
        //     for (int i = 0; i < 4; i++) {
        //         cv::line(frame, bbox[i], bbox[(i + 1) % 4],
        //                  cv::Scalar(0, 255, 0), 2);
        //     }

        //     lib::putTextCentered(frame, decodedData,
        //                          cv::Point2f((bbox[0] + bbox[2]) / 2));
        //     imshow("QRCode", qrcode);
        // }
        if (frame.empty())
            break;
        imshow("Webcam", frame);
        if (cv::waitKey(25) == 'q')  // ESC to quit
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
