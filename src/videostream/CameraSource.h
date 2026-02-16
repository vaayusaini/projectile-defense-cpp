#pragma once

#include <cstdint>
#include <memory>

#include <CoreVideo/CoreVideo.h> // CVPixelBufferRef

namespace pd {

// Owns a CVPixelBufferRef via CFRetain/CFRelease.
// Safe to copy (retains), cheap to move (transfers).
struct ImageFrame {
  uint64_t frame = 0;

  ImageFrame() = default;
  ImageFrame(uint64_t seq, CVPixelBufferRef pb);

  ImageFrame(const ImageFrame &other);
  ImageFrame &operator=(const ImageFrame &other);

  ImageFrame(ImageFrame &&other) noexcept;
  ImageFrame &operator=(ImageFrame &&other) noexcept;

  ~ImageFrame();

  // Non-owning accessor; lifetime is tied to this ImageFrame instance.
  CVPixelBufferRef pixelBuffer() const noexcept { return _pb; }

  // Releases the held buffer (if any) and resets metadata.
  void reset() noexcept;

private:
  CVPixelBufferRef _pb = nullptr;
};

class CameraSource {
public:
  // deviceIndex indexes into [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo]
  explicit CameraSource(int deviceIndex);
  ~CameraSource();

  CameraSource(const CameraSource &) = delete;
  CameraSource &operator=(const CameraSource &) = delete;

  CameraSource(CameraSource &&) noexcept;
  CameraSource &operator=(CameraSource &&) noexcept;

  // Writes the latest frame into `out` (retains safely).
  // Returns false if no frame has arrived yet or capture has stopped/failed.
  bool read(ImageFrame &out);

  // Stops capture; safe to call multiple times.
  void release() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace pd
