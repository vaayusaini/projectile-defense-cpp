#include "position.h"
#include "projectiledetector.h"
#include "projectiletracker.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    testPosition();

    const std::string videoPath = "./samples/basketball.mov";

    cv::VideoCapture firstStream(0);
    cv::VideoCapture secondStream(1);

    std::string firstWindowName = "PD1";
    pd::ProjectileDetector pd1(firstWindowName, firstStream);
    pd1.setDebug(true);

    std::string secondWindowName = "PD2";
    pd::ProjectileDetector pd2(secondWindowName, secondStream);
    pd2.setDebug(true);

    int framesProcessed = 0;
    const int64 startTime = cv::getTickCount();
    std::vector<pd::ProjectileFrame> firstProjectileFrames;
    std::vector<pd::ProjectileFrame> secondProjectileFrames;

    while (true) {
        if (!pd1.process(framesProcessed, firstProjectileFrames)) {
            break;
        }

        if (!pd2.process(framesProcessed, secondProjectileFrames)) {
            break;
        }

        framesProcessed++;
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    const int64 endTime = cv::getTickCount();
    const double secondsElapsed = static_cast<double>(endTime - startTime) / cv::getTickFrequency();

    std::cout << framesProcessed << " frames processed" << std::endl;
    std::cout << secondsElapsed << std::endl;

    firstStream.release();
    secondStream.release();
    cv::destroyAllWindows();

    return 0;
}
