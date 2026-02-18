#include "videostream/CameraSource.h"
#include "videostream/ImageViewer.h"
#include "detector/Detector.h"

#include <chrono>
#include <thread>

int main() {
  pd::CameraSource cam0(0);
  pd::CameraSource cam1(1);
  pd::ImageViewer viewer;

  pd::DetectorConfig detectorCfg;
  detectorCfg.foregroundThreshold = 0.16f;
  detectorCfg.backgroundLearningRate = 0.02f;
  detectorCfg.foregroundUpdateScale = 0.1f;
  detectorCfg.morphologyOpenIterations = 1;
  detectorCfg.morphologyCloseIterations = 1;
  detectorCfg.warmupFrames = 20;
  detectorCfg.minBlobArea = 25;
  detectorCfg.maxBlobArea = 5000;
  detectorCfg.maxBlobAspectRatio = 4.0f;
  detectorCfg.minBlobFillRatio = 0.2f;
  detectorCfg.borderIgnorePixels = 6;
  detectorCfg.maxRawDetections = 32;
  detectorCfg.minConfirmedHits = 3;
  detectorCfg.maxMissedFrames = 5;
  detectorCfg.maxAssociationDistancePx = 55.0f;
  detectorCfg.ballisticGravityY = 0.0f;
  detectorCfg.smoothingAlpha = 0.65f;

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
      const auto points0 = detector0.process(f0);
      pd::drawPoints(f0, points0);
      viewer.show("Camera 0", f0);
      updated0 = true;
    }
    if (ok1 && f1.frame != lastSeq1) {
      lastSeq1 = f1.frame;
      const auto points1 = detector1.process(f1);
      pd::drawPoints(f1, points1);
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
