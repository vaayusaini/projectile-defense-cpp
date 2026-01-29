#include <cmath>
#include <iostream>

// CONSTANTS
// Camera Settings
const double camAngleX = 1.23;
const double camAngleY = 0.96;
const int camResolutionX = 5712;
const int camResolutionY = 4284;

// Camera Positioning relative to 0,0
const double cam1X = 0;
const double cam1Y = 0;
const double cam2X = 45.5;
const double cam2Y = 0;
const double camSeparationX = cam2X - cam1X;
const double camSeparationY = cam2Y - cam1Y;

// Inputs from opencv program
const int cam1PixelX = 3353;
const int cam1PixelY = 1331;
const int cam2PixelX = 1953;
const int cam2PixelY = 1302;

double findXRatio(int pixel) {
    return (std::tan(camAngleX / 2) * (2 * pixel - camResolutionX - 1) / camResolutionX);
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

int testPosition() {
    // Basically finds the value of tan for this angle,
    // except the math is so good I dont need tan
    double xRatio1 = findXRatio(cam1PixelX);
    double yRatio1 = findYRatio(cam1PixelY);
    double xRatio2 = findXRatio(cam2PixelX);
    double yRatio2 = findYRatio(cam2PixelY);

    double zCoord = findDepth(xRatio1, yRatio1, xRatio2, yRatio2);
    double xCoord = (cam1X + cam2X + zCoord * (xRatio1 + xRatio2)) / 2;
    double yCoord = (cam1Y + cam2Y + zCoord * (yRatio1 + yRatio2)) / 2;

    std::cout << "coordinates: (" << xCoord << "," << yCoord << "," << zCoord << ")\n";

    return 0;
}