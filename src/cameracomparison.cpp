#include "cameracomparison.h"
#include "position.h"

namespace pd {

CameraComparison::CameraComparison(ProjectileState* cam1, ProjectileState* cam2): _cam1State(cam1), _cam2State(cam2), coordinates() {};

bool CameraComparison::sameStates(ProjectileState* cam1, ProjectileState* cam2) {
    if ((cam1 == _cam1State) and (cam2 == _cam2State)) {
        return true;
    } 
    return false;
}

void CameraComparison::_addCoordinate(ProjectileFrame* cam1Frame, ProjectileFrame* cam2Frame) {
    if (cam1Frame and cam2Frame) { //if neither one is a nullptr
        std::vector<double> newCoordVector = findCoordinates(cam1Frame->center.x, cam1Frame->center.y, cam2Frame->center.x, cam2Frame->center.y);
        CoordinateFrame newCoord = CoordinateFrame(cam1Frame->frame, newCoordVector[0], newCoordVector[1], newCoordVector[2]);
        coordinates.push_back(newCoord);
    }
}

void CameraComparison::updateCoordinates() {
    for (int i = lastUpdateFrame; i < _cam1State->history.size(); ++i) { //should run from lastUpdateFarme to the end of the history
        _addCoordinate(_cam1State->history[i], _cam2State->history[i]);
    }
    lastUpdateFrame = _cam1State->history.size(); //saying i just updated it
}

}
