#pragma once
#include "projectiletracker.h"

namespace pd {

struct CoordinateFrame {
    int frameNumber;
    int x;
    int y;
    int z;
    CoordinateFrame(const int frame, const int xc, const int yc, const int zc): frameNumber(frame), x(xc), y(yc), z(zc) {};
};

class CameraComparison {
private:
    /* data */
    ProjectileState* _cam1State;
    ProjectileState* _cam2State;
    int lastUpdateFrame = 0;

    void addCoordinate(ProjectileFrame* cam1Frame, ProjectileFrame* cam2Frame);


public:
    CameraComparison(ProjectileState* cam1, ProjectileState* cam2);

    std::vector<CoordinateFrame> coordinates;

    bool sameStates(ProjectileState* cam1S, ProjectileState* cam2S);
    void updateCoordinates();
};

}