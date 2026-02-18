#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace pd {

struct ImageFrame;

struct Pixel {
  int x = 0;
  int y = 0;
};

struct BoundingBox {
  Pixel topLeft;
  Pixel dimensions;
  int area = 0;
};

struct ProjectileFrame {
  Pixel center;
  BoundingBox bbox;
  uint64_t frame = 0;
};

struct DetectorConfig {
  // Resize factor used before bg subtraction / morphology.
  double imscale = 1.0;

  // Background model tuning (MOG2-style names retained for compatibility).
  int bgHistory = 200;
  double varThreshold = 16.0;
  bool detectShadows = false;
  double bgRatio = 0.7;

  // Morphological close config.
  int closeKernelSize = 5;
  int closeIterations = 1;

  // Connected components config.
  int connectivity = 8;
  int minArea = 30;
  float minAspect = 0.25f; // width / height lower bound
  float maxAspect = 4.0f;  // width / height upper bound

  // Optional cap on returned projectiles (largest area first).
  int maxProjectiles = 32;
};

class Detector {
public:
  explicit Detector(const DetectorConfig &config = {});
  ~Detector();

  Detector(const Detector &) = delete;
  Detector &operator=(const Detector &) = delete;

  Detector(Detector &&) noexcept;
  Detector &operator=(Detector &&) noexcept;

  // Updates internal buffers and returns projectile centers for this frame.
  std::vector<Pixel> process(const ImageFrame &frame);

  // Most recent projectile detections with bbox/center metadata.
  const std::vector<ProjectileFrame> &lastProjectiles() const;

  void applyConfig(const DetectorConfig &config);

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

// Draws a filled red point into the frame if writable BGRA data is available.
void drawPoint(ImageFrame &frame, const Pixel &point, int radius = 3);

// Convenience helper to draw multiple points.
void drawPoints(ImageFrame &frame, const std::vector<Pixel> &points, int radius = 3);

// Draws projectile bbox (green) and center (red) on the frame.
void drawProjectiles(ImageFrame &frame, const std::vector<ProjectileFrame> &projectiles, int boxThickness = 2,
                     int centerRadius = 3);

} // namespace pd
