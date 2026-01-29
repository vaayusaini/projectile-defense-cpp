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
    float aspectRatio;
};

struct ProjectileFrame {
    BoundingBox bbox;
    Pixel center;
    int frame;
};

struct ProjectileState {
    int lastUpdateFrame;
    std::vector<ProjectileFrame> history;
};

class ProjectileTracker {
  public:
    ProjectileTracker();
    void checkForPersistentProjectiles(std::vector<ProjectileFrame> frameProjectiles);
    void getPersistentProjectiles(std::vector<ProjectileState> &out);

  private:
    std::vector<ProjectileState> _projectileStates;

    ProjectileState &_getOrCreateProjectileState(ProjectileFrame &projectileFrame);
    void _deleteOldProjectiles();
};

} // namespace pd
