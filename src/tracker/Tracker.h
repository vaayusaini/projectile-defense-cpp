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
  Vector3 _findMiddle(std::vector<Vector3> values, double allowableDistance);
  void _updateTrendline();
  Vector3 _expectedPosition(double t);
  double _domeHeight(double radius);
  void _getIntercept();

public:
  Tracker();
  void print(double t);
  double releaseAngle = 0;
  double releaseTime = 0;
  void updateCoordinates(std::vector<ProjectileFrame> cam1Frames, std::vector<ProjectileFrame> cam2Frames);
};

} // namespace pd