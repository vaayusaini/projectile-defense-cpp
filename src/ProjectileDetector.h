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
    float minAspect = -1;
    float maxAspect = -1;
};

struct Projectile {
    int label = -1;
    int area = 0;
    cv::Rect bbox;        // x,y,w,h
    cv::Point2f centroid; // cx,cy
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
    std::string _name;
    DetectorConfig _config;

    cv::Mat _closeKernel;
    cv::Ptr<cv::BackgroundSubtractorMOG2> _bgSubtractor;
    cv::VideoCapture &_videoStream;

    // reused per-frame buffers (avoid reallocations every frame)
    cv::Mat _raw, _scaled, _fg, _mask;
};

} // namespace pd