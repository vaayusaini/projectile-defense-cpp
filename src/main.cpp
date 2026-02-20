#include "projectiledetector.h"
#include "tracker.h"
#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>



int main() {
    const std::string videoPath1 = "./samples/cam1sample.mov";
    const std::string videoPath2 = "./samples/cam2sample.mov";

    cv::VideoCapture firstStream(videoPath1);
    std::string firstWindowName = "PD1";
    std::vector<pd::ProjectileFrame> firstProjectileFrames;
    pd::ProjectileDetector pd1(firstWindowName, firstStream);
    pd1.setDebug(true);

    cv::VideoCapture secondStream(videoPath2);
    std::string secondWindowName = "PD2";
    std::vector<pd::ProjectileFrame> secondProjectileFrames;
    pd::ProjectileDetector pd2(secondWindowName, secondStream);
    pd2.setDebug(true);

    int framesProcessed = 1;
    const int64 startTime = cv::getTickCount();

    pd::Tracker tracker;

    while (true) {
        // CAMERAS
        if (!pd1.findProjectiles(framesProcessed, firstProjectileFrames)) {
            break;
        }

        if (!pd2.findProjectiles(framesProcessed, secondProjectileFrames)) {
            break;
        }

        // TRACKING
        std::cout << firstProjectileFrames.size();
        std::cout << secondProjectileFrames.size();
        std::cout << framesProcessed;
        std::cout << std::endl;
        tracker.updateCoordinates(firstProjectileFrames, secondProjectileFrames);
        tracker.printV();
        

        framesProcessed ++;
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
