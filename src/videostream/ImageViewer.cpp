#include "ImageViewer.h"
#include "CameraSource.h" // pd::ImageFrame

#include <CoreVideo/CoreVideo.h>

#include <cctype>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/highgui.hpp>

namespace pd {

namespace {

int normalizeKey(int key) {
  if (key < 0)
    return -1;

  const int low = key & 0xFF;
  if (low >= 0 && low <= 255) {
    const unsigned char c = static_cast<unsigned char>(low);
    if (std::isprint(c))
      return std::tolower(c);
  }

  return key;
}

class LockedPixelBufferView {
public:
  explicit LockedPixelBufferView(CVPixelBufferRef pb) noexcept : _pb(pb) {
    if (!_pb)
      return;

    if (CVPixelBufferGetPixelFormatType(_pb) != kCVPixelFormatType_32BGRA)
      return;

    CVPixelBufferLockBaseAddress(_pb, kCVPixelBufferLock_ReadOnly);
    _locked = true;

    void *base = CVPixelBufferGetBaseAddress(_pb);
    if (!base)
      return;

    const int width = static_cast<int>(CVPixelBufferGetWidth(_pb));
    const int height = static_cast<int>(CVPixelBufferGetHeight(_pb));
    const size_t stride = CVPixelBufferGetBytesPerRow(_pb);

    if (width <= 0 || height <= 0 || stride < static_cast<size_t>(width * 4))
      return;

    _mat = cv::Mat(height, width, CV_8UC4, base, stride);
  }

  ~LockedPixelBufferView() {
    if (_locked)
      CVPixelBufferUnlockBaseAddress(_pb, kCVPixelBufferLock_ReadOnly);
  }

  LockedPixelBufferView(const LockedPixelBufferView &) = delete;
  LockedPixelBufferView &operator=(const LockedPixelBufferView &) = delete;

  bool valid() const noexcept { return !_mat.empty(); }
  const cv::Mat &mat() const noexcept { return _mat; }

private:
  cv::Mat _mat;
  CVPixelBufferRef _pb = nullptr;
  bool _locked = false;
};

bool isWindowVisibleNoThrow(const std::string &name) {
  try {
    return cv::getWindowProperty(name, cv::WND_PROP_VISIBLE) >= 1.0;
  } catch (const cv::Exception &) {
    return false;
  }
}

void destroyWindowNoThrow(const std::string &name) noexcept {
  try {
    cv::destroyWindow(name);
  } catch (const cv::Exception &) {
  }
}

} // namespace

struct ImageViewer::Impl {
  mutable std::mutex stateMutex;
  mutable std::mutex highguiMutex;
  std::unordered_map<std::string, cv::Size> windowSizes;
};

ImageViewer::ImageViewer() : _impl(std::make_unique<Impl>()) {}
ImageViewer::~ImageViewer() { closeAll(); }

ImageViewer::ImageViewer(ImageViewer &&other) noexcept = default;
ImageViewer &ImageViewer::operator=(ImageViewer &&other) noexcept {
  if (this == &other)
    return *this;

  closeAll();
  _impl = std::move(other._impl);
  return *this;
}

void ImageViewer::show(const std::string &windowName, const ImageFrame &frame) {
  if (!_impl)
    return;

  LockedPixelBufferView image(frame.pixelBuffer());
  if (!image.valid())
    return;

  const cv::Size currentSize = image.mat().size();

  std::lock_guard<std::mutex> highguiLock(_impl->highguiMutex);

  bool windowKnown = false;
  cv::Size lastSize{};
  {
    std::lock_guard<std::mutex> stateLock(_impl->stateMutex);
    auto it = _impl->windowSizes.find(windowName);
    if (it != _impl->windowSizes.end()) {
      windowKnown = true;
      lastSize = it->second;
    }
  }

  const bool windowVisible = windowKnown ? isWindowVisibleNoThrow(windowName) : false;

  try {
    if (!windowKnown || !windowVisible) {
      cv::namedWindow(windowName, cv::WINDOW_NORMAL);
      if (currentSize.width > 0 && currentSize.height > 0)
        cv::resizeWindow(windowName, currentSize.width, currentSize.height);

      std::lock_guard<std::mutex> stateLock(_impl->stateMutex);
      _impl->windowSizes[windowName] = currentSize;
    } else if (lastSize != currentSize && currentSize.width > 0 && currentSize.height > 0) {
      cv::resizeWindow(windowName, currentSize.width, currentSize.height);

      std::lock_guard<std::mutex> stateLock(_impl->stateMutex);
      auto it = _impl->windowSizes.find(windowName);
      if (it != _impl->windowSizes.end())
        it->second = currentSize;
    }

    cv::imshow(windowName, image.mat());
  } catch (const cv::Exception &) {
    std::lock_guard<std::mutex> stateLock(_impl->stateMutex);
    _impl->windowSizes.erase(windowName);
  }
}

int ImageViewer::waitKey(int delayMs) {
  if (!_impl)
    return -1;

  std::lock_guard<std::mutex> highguiLock(_impl->highguiMutex);
  const int rawKey = (delayMs <= 0) ? cv::pollKey() : cv::waitKeyEx(delayMs);
  return normalizeKey(rawKey);
}

void ImageViewer::close(const std::string &windowName) {
  if (!_impl)
    return;

  {
    std::lock_guard<std::mutex> highguiLock(_impl->highguiMutex);
    destroyWindowNoThrow(windowName);
  }
  {
    std::lock_guard<std::mutex> stateLock(_impl->stateMutex);
    _impl->windowSizes.erase(windowName);
  }
}

void ImageViewer::closeAll() {
  if (!_impl)
    return;

  std::vector<std::string> windowsToClose;
  {
    std::lock_guard<std::mutex> stateLock(_impl->stateMutex);
    windowsToClose.reserve(_impl->windowSizes.size());
    for (const auto &kv : _impl->windowSizes)
      windowsToClose.push_back(kv.first);
    _impl->windowSizes.clear();
  }

  std::lock_guard<std::mutex> highguiLock(_impl->highguiMutex);
  for (const auto &name : windowsToClose)
    destroyWindowNoThrow(name);
}

bool ImageViewer::isOpen(const std::string &windowName) const {
  if (!_impl)
    return false;

  {
    std::lock_guard<std::mutex> stateLock(_impl->stateMutex);
    if (_impl->windowSizes.find(windowName) == _impl->windowSizes.end())
      return false;
  }

  bool visible = false;
  {
    std::lock_guard<std::mutex> highguiLock(_impl->highguiMutex);
    visible = isWindowVisibleNoThrow(windowName);
  }

  if (!visible) {
    std::lock_guard<std::mutex> stateLock(_impl->stateMutex);
    _impl->windowSizes.erase(windowName);
    return false;
  }

  return true;
}

} // namespace pd
