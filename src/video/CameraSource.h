#pragma once
#include <cstdint>
#include <string>

namespace pd {

struct ImageFrame {
    uint64_t frame;
    void *pixelBuffer;
};

class CameraSource {
  public:
    CameraSource(int device);
    virtual bool read(ImageFrame &out) = 0;

  private:
    // hide Obj-C implementation from C++
    struct Impl;
    Impl *_impl;
};

} // namespace pd
