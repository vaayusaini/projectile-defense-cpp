#include "videostream/CameraSource.h"
#include "videostream/ImageViewer.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

struct Latest {
  std::mutex m;
  pd::ImageFrame f0;
  pd::ImageFrame f1;
  bool has0 = false;
  bool has1 = false;
};

int main() {
  pd::CameraSource cam0(0);
  pd::CameraSource cam1(1);

  pd::ImageViewer viewer;

  std::atomic<bool> quit{false};
  Latest latest;

  // Background capture thread: only updates "latest frames"
  std::thread capture([&] {
    pd::ImageFrame tmp0, tmp1;

    while (!quit.load(std::memory_order_relaxed)) {
      bool ok0 = cam0.read(tmp0);
      bool ok1 = cam1.read(tmp1);

      if (ok0 || ok1) {
        std::lock_guard<std::mutex> lock(latest.m);
        if (ok0) {
          latest.f0 = tmp0;
          latest.has0 = true;
        }
        if (ok1) {
          latest.f1 = tmp1;
          latest.has1 = true;
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  });

  // Main/UI loop: pump events + present latest frames
  while (true) {
    // Pull the latest frames (fast lock)
    pd::ImageFrame f0, f1;
    bool has0 = false, has1 = false;
    {
      std::lock_guard<std::mutex> lock(latest.m);
      has0 = latest.has0;
      has1 = latest.has1;
      if (has0)
        f0 = latest.f0;
      if (has1)
        f1 = latest.f1;
    }

    // Present on main thread (NO dispatch flood)
    if (has0)
      viewer.show("Camera 0", f0);
    if (has1)
      viewer.show("Camera 1", f1);

    // Pump UI + read key
    int key = viewer.waitKey(8);
    if (key == 'q')
      break;
  }

  quit.store(true, std::memory_order_relaxed);
  capture.join();

  cam0.release();
  cam1.release();
  viewer.closeAll();
  return 0;
}
