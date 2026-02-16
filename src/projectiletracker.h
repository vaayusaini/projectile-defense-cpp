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
    bool real = true;
    BoundingBox bbox;
    Pixel center;
    int frame;
};

struct ProjectileState {
    int lastUpdateFrame = 0;
    int framesContained = 0;
    std::vector<ProjectileFrame> history;
};

class ProjectileTracker {
  private:
    std::vector<ProjectileState> _projectileStates;

    void _getOrCreateProjectileState(ProjectileFrame newProjectileFrame);
    void _sortAndDeleteOldProjectiles(int currentFrameNumber);

  public:
    ProjectileTracker();

    ProjectileState* getMainProjectile(int currentFrame, std::vector<ProjectileFrame> frameProjectiles);

};

} // namespace pd
