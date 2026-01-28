#pragma once
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/utility.hpp>

namespace pd {

struct ProjectileFrame {
    int area;
    int x; int y; int h; int w;
    int cx; int cy;
    float aspectRatio;
    int frame;
};

struct ProjectileState {
    int lastUpdateFrame;
    std::vector<ProjectileFrame> history;
};

class ProjectileTracker {
    public:
        ProjectileTracker();
        void processProjectile(std::vector<ProjectileFrame> frameProjectiles);
        void getActiveProjectiles(std::vector<ProjectileState>& out);

    private:
        std::vector<ProjectileState> _projectileStates;
        bool _getOrCreateProjectileState(ProjectileFrame& projectileFrame);
        void _addFrameToProjectileState(ProjectileFrame& projectileFrame);
        void _deleteOldProjectiles();
};


}
