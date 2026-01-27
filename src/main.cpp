#include "ProjectileDetector.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {

    const std::string samplesPath = "/Users/vaayusaini/Documents/Local Documents/Projects/Projectile Defense/Samples/";
    const std::string videoPath = samplesPath + "basketball.mov";

    cv::VideoCapture videoStream(videoPath);

    std::string firstWindowName = "PD1";
    pd::ProjectileDetector pd1(firstWindowName, videoStream);

    const int64 startTime = cv::getTickCount();

    pd1.setDebug(true);

    int frames_processed = 0;
    while (true) {
        std::vector<pd::Projectile> projectileLabels;

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