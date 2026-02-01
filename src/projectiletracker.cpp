#include "projectiletracker.h"
#include <cmath>

namespace pd {

ProjectileTracker::ProjectileTracker() : _projectileStates() {};

ProjectileState* ProjectileTracker::_getOrCreateProjectileState(ProjectileFrame* newProjectileFrame) {
    ProjectileState* bestStateMatch = nullptr;
    double bestMatchScore = 2; // i chose this so that (score < bestMatchScore) will return True for the first value 
    for (ProjectileState* ps : _projectileStates) {
        ProjectileFrame* lastFrame = ps->history.back();
        double areaComparison = abs(std::log2(lastFrame->bbox.area / newProjectileFrame->bbox.area)); // if its less than 1 it means the new area is between half and double the old one
        double aspectRatioComparison = abs(std::log2(lastFrame->bbox.aspectRatio / newProjectileFrame->bbox.aspectRatio)); // if its less than 1 it means the new aspectRatio is between half and double the old one
        double score = areaComparison + aspectRatioComparison;
        if ((score < bestMatchScore) and (areaComparison < 1) and (aspectRatioComparison < 1)) {
            bestMatchScore = score;
            bestStateMatch = ps;
        }
    }
    if (bestStateMatch) { //if its not a nullptr
        int missing_slots =  newProjectileFrame->frame - bestStateMatch->lastUpdateFrame;
        for (int i = 0; i < missing_slots; ++i) { //adding a new ProjectileFrame to the end of the State's history
            bestStateMatch->history.emplace_back();
        };
        return bestStateMatch;
    } else {
        ProjectileState* newState = _projectileStates.emplace_back(newProjectileFrame->frame);
        newState->history.push_back(newProjectileFrame);
        return newState;
    }
}

void ProjectileTracker::_sortAndDeleteOldProjectiles(int currentFrameNumber) {
    if (!_projectileStates.empty()) {
        std::sort(_projectileStates.begin(), _projectileStates.end(),
            [](const ProjectileState& a, const ProjectileState& b) {
                if (a.lastUpdateFrame != b.lastUpdateFrame)
                    return a.lastUpdateFrame > b.lastUpdateFrame;
                return a.history.size() > b.history.size();
            });
        for (int i = _projectileStates.size(); i-- > 0; ) {
            if ((currentFrameNumber - _projectileStates[i]->lastUpdateFrame) > 3) {
                _projectileStates.erase(_projectileStates.begin() + i);
            }
        }

    }
}

void ProjectileTracker::checkForPersistentProjectiles(int currentFrame, std::vector<ProjectileFrame*> frameProjectiles) {
    for (ProjectileFrame* p : frameProjectiles) {
        ProjectileState* projectileState = _getOrCreateProjectileState(p);
    }
    _sortAndDeleteOldProjectiles(currentFrame);
}

} // namespace pd