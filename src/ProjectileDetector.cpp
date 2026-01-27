#include "ProjectileDetector.h"

namespace pd {

void closeGaps(cv::InputArray input, cv::OutputArray output, const cv::Mat &kernel, const int iterations) {
    cv::morphologyEx(input, output, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), iterations);
}

ProjectileDetector::ProjectileDetector(std::string name, cv::VideoCapture &videoStream, DetectorConfig config)
    : _name(name), _videoStream(videoStream) {
    if (!_videoStream.isOpened()) {
        throw std::runtime_error("ProjectileProcessor: video stream not opened");
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

    if (_debug) {
        cv::imshow(_name, _scaled);
    }

    return true;
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