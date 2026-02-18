#pragma once

#include <memory>
#include <vector>

namespace pd {

struct ImageFrame;

struct Pixel {
  int x = 0;
  int y = 0;
};

struct DetectorConfig {
  // Foreground threshold in normalized luma [0, 1].
  float foregroundThreshold = 0.16f;

  // Background exponential learning rate in [0, 1].
  float backgroundLearningRate = 0.02f;

  // Extra attenuation applied to learning when pixel is foreground.
  // 0.0 => no bg update on fg pixels, 1.0 => full update.
  float foregroundUpdateScale = 0.1f;

  // Morphological opening iterations (erode -> dilate).
  int morphologyOpenIterations = 1;

  // Morphological closing iterations (dilate -> erode).
  int morphologyCloseIterations = 1;

  // Drop very early frames while background initializes.
  int warmupFrames = 20;

  // Raw connected-component filters.
  int minBlobArea = 25;
  int maxBlobArea = 5000;
  float maxBlobAspectRatio = 4.0f;
  float minBlobFillRatio = 0.2f;
  int borderIgnorePixels = 6;
  int maxRawDetections = 32;

  // Temporal tracking/stability filters.
  float maxAssociationDistancePx = 55.0f;
  int minConfirmedHits = 3;
  int maxMissedFrames = 5;
  float ballisticGravityY = 0.0f;
  float smoothingAlpha = 0.65f;
};

class Detector {
public:
  explicit Detector(const DetectorConfig &config = {});
  ~Detector();

  Detector(const Detector &) = delete;
  Detector &operator=(const Detector &) = delete;

  Detector(Detector &&) noexcept;
  Detector &operator=(Detector &&) noexcept;

  // Runs GPU background subtraction + morphology, then returns stable centers.
  std::vector<Pixel> process(const ImageFrame &frame);

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

// Draws a filled red point into the frame if writable BGRA data is available.
void drawPoint(ImageFrame &frame, const Pixel &point, int radius = 3);

// Convenience helper to draw multiple points.
void drawPoints(ImageFrame &frame, const std::vector<Pixel> &points, int radius = 3);

} // namespace pd

