#include "tracker.h"
#include "position.h"


int fps = 30;
double spf = 1.0 / fps;

namespace pd {
Tracker::Tracker() : _coordinates() {};

void Tracker::updateCoordinates(std::vector<ProjectileFrame> cam1Frames, std::vector<ProjectileFrame> cam2Frames) { //can be run even if there are no frames for one of them
    if ((cam1Frames.size()) != 1 or (cam2Frames.size() != 1)) {
        return;
    }

    int currentFrame = cam1Frames[0].frame;
    if (_coordinates.size() > 0) { //checks that we can validly run the next if statement
        if ((currentFrame - _coordinates.back().frame) > 15) {
            _coordinates.clear();
        }//resets if its been too long since last update
    }
    
    Pixel cam1Center = cam1Frames[0].center;
    Pixel cam2Center = cam2Frames[0].center;
    Vector3 coordinates = findCoordinates(cam1Center.x, cam1Center.y, cam2Center.x, cam2Center.y);
    TrackerFrame newFrame = TrackerFrame(cam1Frames[0].frame, coordinates);
    _coordinates.push_back(newFrame);
}

void Tracker::_updateTrendline() {
    std::vector<Vector3> vIs = {};
    std::vector<Vector3> pIs = {};
    for (size_t i = 0; i < _coordinates.size() / 2; ++i) { //works bc its integer division
        Vector3 c1 = _coordinates[i].position;
        Vector3 displacement = _coordinates[_coordinates.size() / 2 + i].position - c1;
        double t1 = _coordinates[i].frame * spf;
        double t2 = _coordinates[_coordinates.size() / 2 + i].frame * spf;
        double dt = t2 - t1;
        double vIy = displacement.y / dt + 4.9 * dt;
        double v0y = vIy + 9.8 * t1;
        double p0y = c1.y - 0.5 * (v0y + vIy) * (v0y + vIy);
        Vector3 newV(displacement.x / dt, v0y, displacement.z / dt);
        vIs.push_back(newV);
        Vector3 newI(c1.x - newV.x * t1, p0y, c1.z - newV.z * t1);
        pIs.push_back(newI);
    }
    std::sort(vIs.begin(), vIs.end());
    std::sort(pIs.begin(), pIs.end());

    if (vIs.empty()) return;
    //finding the median
    int quarter = round(static_cast<float>(vIs.size()) / 4.0);
    int thirdQuarter = static_cast<int>(vIs.size()) - quarter;
    Vector3 centerVIs;
    Vector3 centerPIs;
    for (int i = quarter; i < thirdQuarter; ++i) {
        centerVIs = centerVIs + vIs[i];
        centerPIs = centerPIs + pIs[i];
    }
    _vI = centerVIs / (thirdQuarter - quarter);
    _pI = centerPIs / (thirdQuarter - quarter);
}

void Tracker::printV() {
    _updateTrendline();
    std::cout << "velocity:" << _vI.x << ", " << _vI.y << ", " << _vI.z << std::endl;
    std::cout << "position:" << _pI.x << ", " << _pI.y << ", " << _pI.z;
}



}