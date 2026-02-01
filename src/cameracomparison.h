#pragma once
#include "projectiletracker.h"

struct CoordinateFrame {
    int x;
    int y;
    int z;
    int vx;
    int vy;
    int vz;
};
namespace pd {
class CameraComparison {
private:
    /* data */
    ProjectileState* _cam1;
    ProjectileState* _cam2;
    std::vector<CoordinateFrame> _coordinates;
public:
    CameraComparison(/* args */);

    bool compare();
    std::vector<int> getTrajectory();
};

}