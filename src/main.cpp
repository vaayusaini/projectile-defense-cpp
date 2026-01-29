#include "position.h"
#include "projectiledetector.h"
#include "projectiletracker.h"
#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>

int main() {
    testPosition();

    const std::string videoPath = "./samples/basketball.mov";

    cv::VideoCapture firstStream(videoPath);
    std::string firstWindowName = "PD1";
    std::vector<pd::ProjectileFrame> firstProjectileFrames;
    pd::ProjectileDetector pd1(firstWindowName, firstStream);
    pd1.setDebug(false);

    cv::VideoCapture secondStream(videoPath);
    std::string secondWindowName = "PD2";
    std::vector<pd::ProjectileFrame> secondProjectileFrames;
    pd::ProjectileDetector pd2(secondWindowName, secondStream);
    pd2.setDebug(false);

    int framesProcessed = 0;
    const int64 startTime = cv::getTickCount();

    while (true) {
        if (!pd1.findProjectiles(framesProcessed, firstProjectileFrames)) {
            break;
        }

        if (!pd2.findProjectiles(framesProcessed, secondProjectileFrames)) {
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
