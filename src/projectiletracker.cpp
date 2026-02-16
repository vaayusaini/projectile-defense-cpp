#include "projectiletracker.h"
#include <cmath>
#include <iostream>

namespace pd {

ProjectileTracker::ProjectileTracker() : _projectileStates() {};

void ProjectileTracker::_getOrCreateProjectileState(ProjectileFrame newProjectileFrame) {
    int bestStateMatch = -1;
    double bestMatchScore = 4; // i chose this so that (score < bestMatchScore) will return True for the first value 
    for (int i = 0; i < _projectileStates.size(); ++i) { 
        ProjectileFrame lastFrame = _projectileStates[i].history.back();
        double areaComparison = abs(std::log2(lastFrame.bbox.area / newProjectileFrame.bbox.area)); // if its less than 1 it means the new area is between half and double the old one
        double aspectRatioComparison = abs(std::log2(lastFrame.bbox.aspectRatio / newProjectileFrame.bbox.aspectRatio)); // if its less than 1 it means the new aspectRatio is between half and double the old one
        double score = areaComparison;
        // if ((score < bestMatchScore) and (areaComparison < 2)) {
        bestMatchScore = score;
        bestStateMatch = i;
        // }
    }
    if (bestStateMatch == -1) { //if its a nullptr, there are no matches so i need to add a new projectileState
        bestStateMatch = _projectileStates.size();
        _projectileStates.push_back(ProjectileState());
    }
    int missing_slots =  newProjectileFrame.frame - 1 - _projectileStates[bestStateMatch].lastUpdateFrame; //the extra -1 is so that we have space to add an extra one at the end, btw vaayu our frames should start counting at 1
    
    for (int i = 0; i < missing_slots; ++i) {
        _projectileStates[bestStateMatch].history.emplace_back(); // adds nullptrs
        _projectileStates[bestStateMatch].history.back().real = false;
    }
    _projectileStates[bestStateMatch].history.push_back(newProjectileFrame);
    _projectileStates[bestStateMatch].framesContained += 1;
    _projectileStates[bestStateMatch].lastUpdateFrame = newProjectileFrame.frame;
}

void ProjectileTracker::_sortAndDeleteOldProjectiles(int currentFrameNumber) {
    if (_projectileStates.empty()) {
        return;
    }
    std::sort(_projectileStates.begin(), _projectileStates.end(),
        [](const ProjectileState& a, const ProjectileState& b) {
            if (a.framesContained != b.framesContained)
                return a.framesContained > b.framesContained;
            return a.lastUpdateFrame > b.lastUpdateFrame;
        });
    for (int i = static_cast<int>(_projectileStates.size()); --i; i >= 0) {
        if ((currentFrameNumber - _projectileStates[i].lastUpdateFrame) > 3) {
            _projectileStates.erase(_projectileStates.begin() + i);
        } 
        else {
            break;
        }
    }
}

ProjectileState* ProjectileTracker::getMainProjectile(int currentFrame, std::vector<ProjectileFrame> frameProjectiles) {
    for (ProjectileFrame p : frameProjectiles) {
        _getOrCreateProjectileState(p);
    }
    _sortAndDeleteOldProjectiles(currentFrame);
    if (_projectileStates.empty()) {
        return nullptr;
    }
    return &(_projectileStates[0]);
}

} // namespace pd