#pragma once
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

struct Projectile {
    int label = -1;
    int area = 0;

    cv::Rect bbox;      // x,y,w,h
    cv::Point2f center; // cx,cy
};

class ProjectileDetector {

  public:
    explicit ProjectileDetector(std::string name, cv::VideoCapture &videoStream, DetectorConfig config = {});
    ~ProjectileDetector() = default;

    bool process(std::vector<Projectile> &projectiles);
    void applyConfig(DetectorConfig config);
    void setDebug(bool debug);

  private:
    bool _debug{};
    DetectorConfig _config;
    std::string _name;

    cv::Mat _closeKernel;
    cv::Ptr<cv::BackgroundSubtractorMOG2> _bgSubtractor;
    cv::VideoCapture &_videoStream;

    // reused per-frame buffers (avoid reallocations every frame)
    cv::Mat _raw, _scaled, _fg, _mask, _labels, _stats, _centroids;

    int extractProjectiles(int numLabels, std::vector<Projectile> &out) const;
};

} // namespace pd