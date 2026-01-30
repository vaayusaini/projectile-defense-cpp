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
    int lastUpdateFrame;
    std::vector<ProjectileFrame*> history;

    ProjectileState(const int frame) : lastUpdateFrame(frame) {};
};

class ProjectileTracker {
  public:
    ProjectileTracker();
    void checkForPersistentProjectiles(int currentFrame, std::vector<ProjectileFrame*> frameProjectiles);
    void getPersistentProjectiles(std::vector<ProjectileState*> out);

  private:
    std::vector<ProjectileState*> _projectileStates;

    ProjectileState* _getOrCreateProjectileState(ProjectileFrame* newProjectileFrame);
    void _sortAndDeleteOldProjectiles(int currentFrameNumber);
};

} // namespace pd
