#include <cmath>
#include <iostream>

// CONSTANTS
// Camera Settings
const double camAngleX = 1.0472;
const double cam1TiltX = 0.174533;
const double cam2TiltX = -0.174533;
const double camAngleY = 0.628319;
const int camResolutionX = 1920;
const int camResolutionY = 1080;

// Camera Positioning relative to 0,0
const double cam1X = -0.5715;
const double cam1Y = 0;
const double cam2X = 0.5715;
const double cam2Y = 0.6096;
const double camSeparationX = cam2X - cam1X;
const double camSeparationY = cam2Y - cam1Y;

double findXRatio(int pixel, double camTilt) {
    return std::tan(std::atan(std::tan(camAngleX / 2) * (2 * pixel - camResolutionX - 1) / camResolutionX) + camTilt);
}

// The difference in math is due to flipping the axis direction of y
double findYRatio(int pixel) {
    return (std::tan(camAngleY / 2) * (camResolutionY - 2 * pixel - 1) / camResolutionY);
}

// Look in Lucas' atpet notebook if you need to understand this
double findDepth(double xR1, double yR1, double xR2, double yR2) {
    double xR = xR1 - xR2;
    double yR = yR1 - yR2;
    return (camSeparationX * xR + camSeparationY * yR) / (xR * xR + yR * yR);
}

std::vector<double> findCoordinates(int cam1PixelX, int cam1PixelY, int cam2PixelX, int cam2PixelY) {
    // Basically finds the value of tan for this angle,
    // except the math is so good I dont need tan
    double xRatio1 = findXRatio(cam1PixelX, cam1TiltX);
    double yRatio1 = findYRatio(cam1PixelY);
    double xRatio2 = findXRatio(cam2PixelX, cam2TiltX);
    double yRatio2 = findYRatio(cam2PixelY);

    double zCoord = findDepth(xRatio1, yRatio1, xRatio2, yRatio2);
    double xCoord = (cam1X + cam2X + zCoord * (xRatio1 + xRatio2)) / 2;
    double yCoord = (cam1Y + cam2Y + zCoord * (yRatio1 + yRatio2)) / 2;

    std::cout << "coordinates: (" << xCoord << "," << yCoord << "," << zCoord << ")\n";
    std::vector<double> coords = {xCoord, yCoord, zCoord};
    return coords;
}