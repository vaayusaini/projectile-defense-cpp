#pragma once
#include "projectiletracker.h"
#include <opencv2/opencv.hpp>

namespace pd {

struct DetectorConfig {
    double imscale = 0.5;

    int bgHistory = 16;
    bool detectShadows = false;
    double varThreshold = 512;
    double bgRatio = 0.978;

    int closeKernelSize = 16;
    int closeIterations = 2;

    int minArea = 1400;
    int connectivity = 8;
    float minAspect = 0;
    float maxAspect = 10;
};

class ProjectileDetector {

  public:
    explicit ProjectileDetector(std::string name, cv::VideoCapture& videoStream, DetectorConfig config = {});
    ~ProjectileDetector() = default;

    bool process(const int frame, std::vector<ProjectileFrame>& projectiles);
    void applyConfig(DetectorConfig config);
    void setDebug(bool debug);

  private:
    bool _debug{};
    DetectorConfig _config;
    std::string _name;

    cv::Mat _closeKernel;
    cv::Ptr<cv::BackgroundSubtractorMOG2> _bgSubtractor;
    cv::VideoCapture& _videoStream;

    // reused per-frame buffers (avoid reallocations every frame)
    cv::Mat _raw, _scaled, _fg, _mask, _labels, _stats, _centroids;

    int _extractProjectiles(int numLabels, int frame, std::vector<ProjectileFrame>& out) const;
};

} // namespace pd
