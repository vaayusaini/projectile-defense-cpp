#include "cameracomparison.h"
#include "movement.h"
#include "projectile.h"
#include "projectiledetector.h"
#include "projectiletracker.h"
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

    pd::ProjectileTracker pt1 = pd::ProjectileTracker();
    pd::ProjectileTracker pt2 = pd::ProjectileTracker();
    pd::CameraComparison workingCameraComparison = pd::CameraComparison(nullptr, nullptr);
    pd::Projectile workingProjectile = pd::Projectile(workingCameraComparison.coordinates);
    
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

        pd::ProjectileState* mainProjectileState1 = pt1.getMainProjectile(framesProcessed, firstProjectileFrames);
        pd::ProjectileState* mainProjectileState2 = pt2.getMainProjectile(framesProcessed, secondProjectileFrames);

        if (framesProcessed == 520) {
            std::cout << "hi";
        }

        if (!workingCameraComparison.sameStates(mainProjectileState1, mainProjectileState2)) {
            workingCameraComparison = pd::CameraComparison(mainProjectileState1, mainProjectileState2);
            pd::Projectile workingProjectile = pd::Projectile(workingCameraComparison.coordinates);
        }
        workingCameraComparison.updateCoordinates();
        workingProjectile.getIntercept();
        // std::cout << "angle " << workingProjectile.releaseAngle << std::endl;
        // std::cout << "time " << workingProjectile.releaseTime << std::endl;
        // //Figure out Motor movement here
        // if ((workingProjectile.releaseTime - (framesProcessed/30)) < 0.03) { //the 30 and the 0.03 will change if we change fps
        //     while ((workingProjectile.releaseTime - (framesProcessed/30)) < 0.03) {
                
        //     }
        //     // TRIGGER HERE

        //     break;
        // }

        framesProcessed ++;
        if (cv::waitKey(1) == 'q') {
            break;
        }

        if (framesProcessed == 30) {
            std::cout << (mainProjectileState1->lastUpdateFrame);
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
