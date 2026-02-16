#pragma once

#include <memory>
#include <string>

namespace pd {

struct ImageFrame;

class ImageViewer {
public:
  ImageViewer();
  ~ImageViewer();

  ImageViewer(const ImageViewer &) = delete;
  ImageViewer &operator=(const ImageViewer &) = delete;

  ImageViewer(ImageViewer &&) noexcept;
  ImageViewer &operator=(ImageViewer &&) noexcept;

  // High-FPS: updates "latest frame" for the named window (drops intermediates).
  // Auto-creates window on first call. Window auto-resizes to the frame size.
  void show(const std::string &windowName, const ImageFrame &frame);

  // Explicit event pumping like OpenCV.
  // Returns:
  //   - 'q' for q or Q (easy loop exit)
  //   - other printable keys as lowercase ascii when possible
  //   - -1 if no key pressed
  int waitKey(int delayMs = 1);

  void close(const std::string &windowName);
  void closeAll();

  bool isOpen(const std::string &windowName) const;

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace pd
