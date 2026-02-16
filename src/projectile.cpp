#include "projectile.h"
#include <iostream>

namespace pd {


double gunYAngle = 0.261799; //in radians
double gunVelocity = 36.576; //in meters per second

double v0yd = gunVelocity * std::sin(gunYAngle);
double v0rd = gunVelocity * std::cos(gunYAngle);

double boxPlot(std::vector<double> v) {
    size_t n = v.size();
    double total = 0;
    int lower = std::floor(n / 4);
    int upper = std::ceil(3 * n / 4);
    for (int i = lower; i < upper; ++i) {
        total += v[i];
    }
    return (total / (upper - lower));
}

Projectile::Projectile(std::vector<CoordinateFrame>& coords): _coordinates(coords) {};

void Projectile::_updateTrendlineVectors(CoordinateFrame firstFrame, CoordinateFrame secondFrame) {
    int framesElapsed = secondFrame.frameNumber - firstFrame.frameNumber;
    double timeElapsed = (1/30) * framesElapsed;
    double currentTime = (1 / 30) * secondFrame.frameNumber;

    double latestXVelocity = (secondFrame.x - firstFrame.x) / timeElapsed; //the 1/30 converts frames into seconds
    double latestZVelocity = (secondFrame.z - firstFrame.z) / timeElapsed; //the 1/30 converts frames into seconds

    double latestYVelocity = (secondFrame.y - firstFrame.y) / timeElapsed;
    double latestStartingYVelocity = latestYVelocity + 4.9 * ((1/30) * (secondFrame.frameNumber + firstFrame.frameNumber)); //look in lucas' atpet notebook if you need to understand this
    double latestStartingYPosition = 4.9 * currentTime * currentTime - latestStartingYVelocity * currentTime + secondFrame.y;
    
    for (int i = 0; i < framesElapsed; ++i) {
        _xVelocities.push_back(latestXVelocity);
        _zVelocities.push_back(latestZVelocity);
        _yVelocities.push_back(latestStartingYVelocity);

        _xInitials.push_back(secondFrame.x - latestXVelocity * currentTime);
        _zInitials.push_back(secondFrame.z - latestZVelocity * currentTime);
        _yInitials.push_back(latestStartingYPosition);
    }
    std::sort(_xVelocities.begin(), _xVelocities.end());
    std::sort(_zVelocities.begin(), _zVelocities.end());
    std::sort(_yVelocities.begin(), _yVelocities.end());

    std::sort(_xInitials.begin(), _xInitials.end());
    std::sort(_zInitials.begin(), _zInitials.end());
    std::sort(_yInitials.begin(), _yInitials.end());
}

void Projectile::_updateTrendline() {
    for (int i = _coordinatesProcessed; _coordinates.size(); ++i) {
        _updateTrendlineVectors(_coordinates[i - 1], _coordinates[i]);
    }

    _vx = boxPlot(_xVelocities);
    _vy = boxPlot(_yVelocities);
    _vz = boxPlot(_zVelocities);

    _x0 = boxPlot(_xInitials);
    _y0 = boxPlot(_yInitials);
    _z0 = boxPlot(_zInitials);

    _coordinatesProcessed = _coordinates.size();
}

double Projectile::_xPosition(double time) {
    return (_vx * time + _x0);
}

double Projectile::_zPosition(double time) {
    return (_vz * time + _z0);
}

double Projectile::_yPosition(double time) {
    return (-4.9 * time * time + _vy * time + _y0);
}

double Projectile::_yAsAFuncOfR(double t) {
    double rp = std::sqrt(_xPosition(t) * _xPosition(t) + _zPosition(t) * _zPosition(t));
    return (-4.9 / (v0yd * v0yd) * (rp * rp) + (v0yd / v0rd) * rp);
}

void Projectile::_calculateIntercept() {

    double t = _coordinates.back().frameNumber / 30; // no point starting our t earlier than now
    while (_yPosition(t) > _yAsAFuncOfR(t)) {
        t += 0.05;
        if (_yPosition(t) < 0) {
            std::cout << "no intercept found";
        }
    }

    releaseAngle = std::atan(_xPosition(t) / _zPosition(t)); // in radians
    releaseTime = t - std::sqrt(_xPosition(t) + _zPosition(t)) / v0rd; // in seconds
}

void Projectile::getIntercept() {
    if (_coordinates.size() > 1) {
        _updateTrendline();
        _calculateIntercept();
    }
}

}

