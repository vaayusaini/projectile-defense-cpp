#pragma once
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/utility.hpp>

namespace pd {

struct Pixel {
    int x;
    int y;
};

struct BoundingBox {
    Pixel topLeft;
    Pixel dimensions;

    int area;
    double aspectRatio;
};

struct ProjectileFrame {
    BoundingBox bbox;
    Pixel center;
    int frame;
};

struct ProjectileState {
    int lastUpdateFrame = 0;
    std::vector<ProjectileFrame*> history;
};

class ProjectileTracker {
  private:
    std::vector<ProjectileState> _projectileStates;

    ProjectileState* _getOrCreateProjectileState(ProjectileFrame* newProjectileFrame);
    void _sortAndDeleteOldProjectiles(int currentFrameNumber);

  public:
    ProjectileTracker();

    void checkForPersistentProjectiles(int currentFrame, std::vector<ProjectileFrame*> frameProjectiles);
    void getPersistentProjectiles(std::vector<ProjectileState*> out);

};

} // namespace pd
