#include "detector/Detector.h"
#include "videostream/CameraSource.h"
#include "videostream/ImageViewer.h"

#include <chrono>
#include <thread>

int main() {
  pd::CameraSource cam0(0);
  pd::CameraSource cam1(1);
  pd::ImageViewer viewer;

  pd::DetectorConfig detectorCfg;
  detectorCfg.imscale = 0.5;
  detectorCfg.bgHistory = 120;
  detectorCfg.varThreshold = 16.0;
  detectorCfg.detectShadows = false;
  detectorCfg.bgRatio = 0.98;
  detectorCfg.closeKernelSize = 3;
  detectorCfg.closeIterations = 1;
  detectorCfg.connectivity = 8;
  detectorCfg.minArea = 300;
  detectorCfg.minAspect = 0.35f;
  detectorCfg.maxAspect = 2.8f;
  detectorCfg.maxProjectiles = 6;

  pd::Detector detector0(detectorCfg);
  pd::Detector detector1(detectorCfg);

  pd::ImageFrame f0;
  pd::ImageFrame f1;
  uint64_t lastSeq0 = 0;
  uint64_t lastSeq1 = 0;

  while (true) {

    const bool ok0 = cam0.read(f0);
    const bool ok1 = cam1.read(f1);
    bool updated0 = false;
    bool updated1 = false;

    if (ok0 && f0.frame != lastSeq0) {
      lastSeq0 = f0.frame;
      detector0.process(f0);
      pd::drawProjectiles(f0, detector0.lastProjectiles());
      viewer.show("Camera 0", f0);
      updated0 = true;
    }
    if (ok1 && f1.frame != lastSeq1) {
      lastSeq1 = f1.frame;
      detector1.process(f1);
      pd::drawProjectiles(f1, detector1.lastProjectiles());
      viewer.show("Camera 1", f1);
      updated1 = true;
    }

    const int key = viewer.waitKey(8);
    if (key == 'q')
      break;

    if (!updated0 && !updated1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  return 0;
}
