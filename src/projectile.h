#pragma once
#include "cameracomparison.h"


namespace pd {

class Projectile {
private:
    int _coordinatesProcessed = 1; //this can be used to determine where to start the next iteration of velocity calculations, and its 1 bc of the for loop in _updateVelocities

    double _vx;
    std::vector<double> _xVelocities;
    double _vy; //this approximates the velocity at frame 0
    std::vector<double> _yVelocities; //approximate velocities at frame 0
    double _vz;
    std::vector<double> _zVelocities;

    double _x0;
    std::vector<double> _xInitials;
    double _y0;
    std::vector<double> _yInitials;
    double _z0;
    std::vector<double> _zInitials;

    std::vector<CoordinateFrame>& _coordinates;

    void _updateTrendlineVectors(CoordinateFrame firstFrame, CoordinateFrame secondFrame);
    void _updateTrendline();

    double _xPosition(double time);
    double _zPosition(double time);
    double _yPosition(double time);

    double _yAsAFuncOfR(double t);

    void _calculateIntercept();
public:
    Projectile(std::vector<CoordinateFrame>& coords);

    void getIntercept();
    double releaseAngle;
    double releaseTime;
};



}
