#pragma once
#include "../detector/Detector.h"
#include "Vector3.h"
#include <vector>

namespace pd {

struct TrackerFrame {
  int frame;
  Vector3 position;
  TrackerFrame(int f, Vector3 coords) : position(coords), frame(f) {}
};

class Tracker {
private:
  Vector3 _vI; // initial velocities
  Vector3 _pI; // initial position
  std::vector<TrackerFrame> _coordinates;
  void _updateVelocities();
  void _updateInitialPositions();
  void _updateTrendline();

public:
  Tracker();
  void printV(double t);
  double releaseAngle;
  double releaseTime;
  void updateCoordinates(std::vector<ProjectileFrame> cam1Frames, std::vector<ProjectileFrame> cam2Frames);
  void getIntercept();
};

} // namespace pd