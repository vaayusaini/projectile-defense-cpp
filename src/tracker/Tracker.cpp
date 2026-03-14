#include "Tracker.h"
#include "Position.h"
#include <iostream>
#include <cmath>

int fps = 30;
double spf = 1.0 / fps;
double longestAllowableCoordinateGap = 10;
double velocityOutlierDelta = 10.0;
double positionOutlierDistance = 5.0;

double triggerDelay = 0.333;
double gunVelocity = 36.576;
double gunAngle = 0.35; //this is just a placeholder cause we havent tested it
double gv0R = gunVelocity * cos(gunAngle);
double gv0Y = gunVelocity * sin(gunAngle);
// these two could be eliminated (go to 0) if we decrease the trigger time and make the height at the pivot point y = 0
double dartReleaseHeight = 0.1016; // the height of where the dart leaves the front of the gun
double horizontalReleaseShift = 0.1524; //the horizontal distance from the front of the gun to the pivot point


namespace pd {
Tracker::Tracker() : _coordinates() {};

void Tracker::updateCoordinates( //gets called from main
    std::vector<ProjectileFrame> cam0Frames,
    std::vector<ProjectileFrame> cam1Frames) { // can be run even if there are no frames for one of them
  if ((cam0Frames.size()) != 1 or (cam1Frames.size() != 1)) {
    return;
  }

  int currentFrame = cam0Frames[0].frame;
  if (_coordinates.size() > 0) { // checks that we can validly run the next if statement
    if ((currentFrame - _coordinates.back().frame) > longestAllowableCoordinateGap) {
      _coordinates.clear();
      _vI = Vector3(0, 0, 0);
      _pI = Vector3(0, 0, 0);
    } // resets if its been too long since last update
  }

  Pixel cam1Center = cam0Frames[0].center;
  Pixel cam2Center = cam1Frames[0].center;
  Vector3 coordinates = findCoordinates(cam1Center.x, cam1Center.y, cam2Center.x, cam2Center.y);
  TrackerFrame newFrame = TrackerFrame(cam0Frames[0].frame, coordinates);
  _coordinates.push_back(newFrame);
  if (_coordinates.size() > 1) {
    _updateTrendline();
    _getIntercept();
  }
}

Vector3 Tracker::_findMiddle(std::vector<Vector3> values, double allowableDistance) { //gets called in _updateTrendline
  std::vector<double> xs, ys, zs;
  for (const Vector3& value : values) {
    xs.push_back(value.x);
    ys.push_back(value.y);
    zs.push_back(value.z);
  }
  std::sort(xs.begin(), xs.end());
  std::sort(ys.begin(), ys.end());
  std::sort(zs.begin(), zs.end());
  //REMOVING OUTLIERS
  int middleIndex = static_cast<int>(values.size() / 2); // not quite if it has an odd size but thats not really important
  Vector3 median(xs[middleIndex], ys[middleIndex], zs[middleIndex]);
  std::vector<Vector3> usable;
  for (int i = 0; i < values.size(); ++i) {
    Vector3 diff = median - values[i];
    if (diff.magnitudeSquared() < allowableDistance * allowableDistance) {
      usable.push_back(values[i]);
    }
  }
  //AVERAGING
  Vector3 sums;
  for (int i = 0; i < usable.size(); ++i) {
    sums += usable[i];
  }
  return (sums / usable.size());
}

void Tracker::_updateTrendline() { //gets called when coordinates are updated and the size is greater than 1
  std::vector<Vector3> vIs = {};
  std::vector<Vector3> pIs = {};
  for (size_t i = 0; i < _coordinates.size() / 2; ++i) { // works bc its integer division
    int i2 = _coordinates.size() / 2 + i;

    Vector3 c1 = _coordinates[i].position;
    Vector3 displacement = _coordinates[i2].position - c1;

    double t1 = _coordinates[i].frame * spf;
    double t2 = _coordinates[i2].frame * spf;
    double dt = t2 - t1;

    double vFy = displacement.y / dt + 4.9 * dt; // the velocity at t1
    double v0y = vFy + 9.8 * t1; // the velocity at time 0
    double p0y = c1.y - 0.5 * (v0y + vFy) * t1; // the position at time 0

    Vector3 newV(displacement.x / dt, v0y, displacement.z / dt);
    vIs.push_back(newV);

    Vector3 newI(c1.x - newV.x * t1, p0y, c1.z - newV.z * t1);
    pIs.push_back(newI);
  }
  _vI = _findMiddle(vIs, velocityOutlierDelta);
  _pI = _findMiddle(pIs, positionOutlierDistance);
}

Vector3 Tracker::_expectedPosition(double t) {
  return Vector3(_vI.x * t + _pI.x, -4.9 * t * t + _vI.y * t + _pI.y, _vI.z * t + _pI.z);
}

double Tracker::_domeHeight(double radius) {
  double redefinedRadius = radius - horizontalReleaseShift;
  double dartT = radius / gv0R; // time it would take for the dart to get to this radius
  return (-4.9 * dartT * dartT + gv0Y * dartT + dartReleaseHeight); // finding the height using that time
}

void Tracker::_getIntercept() { //called if coordinates are updated
  double t = _coordinates.back().frame * spf;
  Vector3 iteratedPosition = _expectedPosition(t);
  if (iteratedPosition.y > _domeHeight(std::sqrt(iteratedPosition.x * iteratedPosition.x + iteratedPosition.z * iteratedPosition.z))) {
    while (iteratedPosition.y > _domeHeight(std::sqrt(iteratedPosition.x * iteratedPosition.x + iteratedPosition.z * iteratedPosition.z))) {
      t += spf; // this step could be different this just seemed like a reasonable number
      if (iteratedPosition.y < 0) {
        return;
      }
    }
    releaseTime = t - triggerDelay;
    releaseAngle = std::atan(iteratedPosition.x / iteratedPosition.z);
    std::cout << "intercept location: " << _expectedPosition(t) << std::endl;
  }

}

void Tracker::print(double t) {
  std::cout << "expected position: " << _expectedPosition(t) << std::endl;
  std::cout << "starting position: " << _pI << std::endl;
  std::cout << "releaseAngle: " << releaseAngle << std::endl;
  std::cout << "releaseTime: " << releaseTime << std::endl;
}

} // namespace pd