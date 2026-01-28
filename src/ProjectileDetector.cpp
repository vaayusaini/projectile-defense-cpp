#include "projectiledetector.h"
#include <opencv2/opencv.hpp>

namespace pd {

void closeGaps(cv::InputArray input, cv::OutputArray output, const cv::Mat &kernel, const int iterations) {
    cv::morphologyEx(input, output, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), iterations);
}

template <typename V, typename L, typename H> bool isWithinBounds(V value, L minVal, H maxVal) {
    return value >= minVal && value <= maxVal;
}

const cv::Scalar greenColor = cv::Scalar(0, 255, 0);
const cv::Scalar redColor = cv::Scalar(0, 0, 255);

void drawProjectilesOnImage(cv::Mat &image, const std::vector<Projectile> &projectiles) {
    for (const Projectile &p : projectiles) {
        // Bounding box
        cv::rectangle(image, p.bbox, greenColor, 2);

        // Centroid
        cv::Point pixelCenter = cv::Point(static_cast<int>(p.center.x), static_cast<int>(p.center.y));
        cv::circle(image, pixelCenter, 3, redColor, cv::FILLED);
    }
}

ProjectileDetector::ProjectileDetector(std::string name, cv::VideoCapture &videoStream, DetectorConfig config)
    : _name(name), _videoStream(videoStream) {
    if (!_videoStream.isOpened()) {
        throw std::runtime_error("ProjectileDetector: video stream not opened");
    }
    applyConfig(config);
}

bool ProjectileDetector::process(std::vector<Projectile> &projectiles) {
    _videoStream.read(_raw);
    if (_raw.empty()) {
        return false;
    }

    // Resize raw image
    cv::resize(_raw, _scaled, cv::Size(), _config.imscale, _config.imscale, cv::INTER_LINEAR);

    // Apply background subtraction
    _bgSubtractor->apply(_scaled, _fg);

    // Close gaps in the foreground mask
    closeGaps(_fg, _mask, _closeKernel, _config.closeIterations);

    // Get projectile labels
    int numLabels = cv::connectedComponentsWithStats(_mask, _labels, _stats, _centroids, _config.connectivity, CV_32S);

    // Extract projectiles to output vector
    extractProjectiles(numLabels, projectiles);

    if (_debug) {
        drawProjectilesOnImage(_scaled, projectiles);
        cv::imshow(_name, _scaled);
    }

    return true;
}

int ProjectileDetector::extractProjectiles(const int numLabels, std::vector<Projectile> &out) const {
    out.clear();

    if (numLabels <= 1)
        return 0;

    // Reserve worst-case (all labels except background). Doesn’t shrink capacity.
    out.reserve(static_cast<size_t>(numLabels - 1));

    // stats: numLabels x 5 (CV_32S), centroids: numLabels x 2 (CV_64F)
    for (int label = 1; label < numLabels; ++label) {
        const int *s = _stats.ptr<int>(label);           // [LEFT, TOP, WIDTH, HEIGHT, AREA]
        const double *c = _centroids.ptr<double>(label); // [CX, CY]

        const int x = s[cv::CC_STAT_LEFT];
        const int y = s[cv::CC_STAT_TOP];
        const int w = s[cv::CC_STAT_WIDTH];
        const int h = s[cv::CC_STAT_HEIGHT];
        const int area = s[cv::CC_STAT_AREA];

        if (w == 0 || h == 0)
            continue;
        if (area < _config.minArea)
            continue;

        const float aspect = static_cast<float>(w) / static_cast<float>(h);
        if (aspect < _config.minAspect || aspect > _config.maxAspect)
            continue;

        Projectile &p = out.emplace_back();
        p.label = label;
        p.area = area;
        p.bbox = cv::Rect(x, y, w, h);
        p.center = cv::Point2f(static_cast<float>(c[0]), static_cast<float>(c[1]));
    }

    return static_cast<int>(out.size());
}

void ProjectileDetector::setDebug(const bool debug) {
    _debug = debug;
}

void ProjectileDetector::applyConfig(DetectorConfig config) {
    _config = config;

    _bgSubtractor = cv::createBackgroundSubtractorMOG2(_config.bgHistory, _config.varThreshold, _config.detectShadows);

    // Update background subtractor parameters
    _bgSubtractor->setHistory(_config.bgHistory);
    _bgSubtractor->setDetectShadows(_config.detectShadows);
    _bgSubtractor->setVarThreshold(_config.varThreshold);
    _bgSubtractor->setBackgroundRatio(_config.bgRatio);

    // Update morphological kernel
    _closeKernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(_config.closeKernelSize, _config.closeKernelSize));
}

} // namespace pd
