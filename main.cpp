#include <getopt.h>

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;

#define RECT_SIZE 10

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        std::vector<Point2i>* pointsVector =
            static_cast<std::vector<Point2i>*>(userdata);
        pointsVector->emplace_back(x, y);
    }
}

void drawRectangleAround(Mat& frame, const Point2i& point) {
    int x = point.x;
    int y = point.y;
    Point leftTop(std::max(0, std::min(x - RECT_SIZE, frame.cols)),
                  std::max(0, std::min(y - RECT_SIZE, frame.rows)));

    Point rightBottom(std::max(0, std::min(x + RECT_SIZE, frame.cols)),
                      std::max(0, std::min(y + RECT_SIZE, frame.rows)));
    rectangle(frame, leftTop, rightBottom, Scalar(0, 255, 0), 2);
}

int main(int argc, char* argv[]) {
    int opt;
    int cameraDev;
    const char* options = "c:";

    while ((opt = getopt(argc, argv, options)) != -1) {
        switch (opt) {
            case 'c': cameraDev = std::stoi(optarg); break;
            default:
                std::cerr << "Error dureing option parsing" << std::endl;
                return EXIT_FAILURE;
        }
    }

    VideoCapture cap(cameraDev);
    if (!cap.isOpened()) {
        printf("Error: Cannot open camera\n");
        return -1;
    }
    Mat frame;
    std::vector<Point2i> pointsToDraw;
    pointsToDraw.reserve(16);

    namedWindow("Webcam");
    setMouseCallback("Webcam", onMouse, &pointsToDraw);
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;
        for (auto& p : pointsToDraw)
            drawRectangleAround(frame, p);
        imshow("Webcam", frame);
        if(waitKey(25) == 'c')
            pointsToDraw.clear();
        if (waitKey(25) == 'q')  // ESC to quit
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
