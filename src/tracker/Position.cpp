#include "Position.h"
#include <cmath>
#include <iostream>

namespace pd {
// CONSTANTS
// Camera Settings
const double camAngleX = 1.047;
const double cam0TiltX = 0.5; // 0.6435; // 0.15;
const double cam1TiltX = -0.195;
const double camAngleY = 0.523;  // 0.628;  // 0.547;  // 0.628319;
const int camResolutionX = 1920; // make sure these match vaayu's imscale
const int camResolutionY = 1080;

// Camera Positioning relative to 0,0
const double cam0X = -3.007; //-0.5874;
const double cam0Y = 0;
const double cam1X = 0.5874;
const double cam1Y = 0.6096;
const double camSeparationX = cam1X - cam0X;
const double camSeparationY = cam1Y - cam0Y;

double findXRatio(int pixel, double camTilt) {
  return std::tan(std::atan(std::tan(camAngleX / 2) * (2 * pixel - camResolutionX - 1) / camResolutionX) + camTilt);
}

// The difference in math is due to flipping the axis direction of y
double findYRatio(int pixel) { return (std::tan(camAngleY / 2) * (camResolutionY - 2 * pixel - 1) / camResolutionY); }

// Look in Lucas' atpet notebook if you need to understand this
double findDepth(double xR1, double yR1, double xR2, double yR2) {
  double xR = xR1 - xR2;
  double yR = yR1 - yR2;
  return (camSeparationX * xR + camSeparationY * yR) / (xR * xR + yR * yR);
}

Vector3 findCoordinates(int cam0PixelX, int cam0PixelY, int cam1PixelX, int cam1PixelY) {
  // Basically finds the value of tan for this angle,
  // except the math is so good I dont need tan
  double xRatio1 = findXRatio(cam0PixelX, cam0TiltX);
  double yRatio1 = findYRatio(cam0PixelY);
  double xRatio2 = findXRatio(cam1PixelX, cam1TiltX);
  double yRatio2 = findYRatio(cam1PixelY);

  double zCoord = findDepth(xRatio1, yRatio1, xRatio2, yRatio2);
  double xCoord = (cam0X + cam1X + zCoord * (xRatio1 + xRatio2)) / 2;
  double yCoord = (cam0Y + cam1Y + zCoord * (yRatio1 + yRatio2)) / 2;

  std::cout << "coordinates: (" << xCoord << "," << yCoord << "," << zCoord << ")\n";
  Vector3 coords(xCoord, yCoord, zCoord);
  return coords;
}

} // namespace pd