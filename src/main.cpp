#include "ProjectileDetector.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "position.h"

int main() {
    testPosition();
    const std::string samplesPath = "/Users/vaayusaini/Documents/Local Documents/Projects/Projectile Defense/Samples/";
    const std::string videoPath = samplesPath + "basketball.mov";

    cv::VideoCapture videoStream(0);

    std::string firstWindowName = "PD1";
    pd::ProjectileDetector pd1(firstWindowName, videoStream);
    pd1.setDebug(true);

    int frames_processed = 0;
    const int64 startTime = cv::getTickCount();
    std::vector<pd::Projectile> projectileLabels;

    while (true) {
        if (!pd1.process(projectileLabels)) {
            break;
        }

        frames_processed++;
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    const int64 endTime = cv::getTickCount();
    const double secondsElapsed = static_cast<double>(endTime - startTime) / cv::getTickFrequency();

    std::cout << frames_processed << " frames processed" << std::endl;
    std::cout << secondsElapsed << std::endl;

    videoStream.release();
    cv::destroyAllWindows();

    return 0;
}