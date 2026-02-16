#import "CameraSource.h"

#import <AVFoundation/AVFoundation.h>
#import <CoreFoundation/CoreFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>

#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <utility>

#pragma mark - ObjC delegate (global namespace)

// No C++ types here. Just an opaque context pointer + callback.
@interface PDFrameDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(nonatomic, assign) void *ctx;
@property(nonatomic, assign) void (*onSample)(void *ctx, CMSampleBufferRef sample);
@end

@implementation PDFrameDelegate
- (void)captureOutput:(AVCaptureOutput *)captureOutput
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection {
  (void)captureOutput;
  (void)connection;
  if (self.onSample)
    self.onSample(self.ctx, sampleBuffer);
}
@end

namespace pd {

// ---------------- ImageFrame RAII ----------------

static inline CVPixelBufferRef RetainPB(CVPixelBufferRef pb) noexcept {
  if (pb)
    CFRetain(pb);
  return pb;
}
static inline void ReleasePB(CVPixelBufferRef pb) noexcept {
  if (pb)
    CFRelease(pb);
}

ImageFrame::ImageFrame(uint64_t seq, CVPixelBufferRef pb) : frame(seq), _pb(RetainPB(pb)) {}

ImageFrame::ImageFrame(const ImageFrame &other) : frame(other.frame), _pb(RetainPB(other._pb)) {}

ImageFrame &ImageFrame::operator=(const ImageFrame &other) {
  if (this == &other)
    return *this;
  reset();
  frame = other.frame;
  _pb = RetainPB(other._pb);
  return *this;
}

ImageFrame::ImageFrame(ImageFrame &&other) noexcept : frame(other.frame), _pb(other._pb) {
  other.frame = 0;
  other._pb = nullptr;
}

ImageFrame &ImageFrame::operator=(ImageFrame &&other) noexcept {
  if (this == &other)
    return *this;
  reset();
  frame = other.frame;
  _pb = other._pb;
  other.frame = 0;
  other._pb = nullptr;
  return *this;
}

ImageFrame::~ImageFrame() { reset(); }

void ImageFrame::reset() noexcept {
  ReleasePB(_pb);
  _pb = nullptr;
  frame = 0;
}

// ---------------- CameraSource::Impl ----------------

namespace {
struct LatestFrame {
  uint64_t seq = 0;
  CVPixelBufferRef pb = nullptr; // retained while stored
};

static void *const kCaptureQueueKey = (void *)&kCaptureQueueKey;

static void DebugPrintDevices(NSArray<AVCaptureDevice *> *devices) {
  std::fprintf(stderr, "[pd] Discovered %lu camera device(s):\n", (unsigned long)devices.count);
  for (NSUInteger i = 0; i < devices.count; ++i) {
    AVCaptureDevice *d = devices[i];
    std::fprintf(stderr, "  [%lu] %s  (uniqueID=%s, type=%s)\n", (unsigned long)i, d.localizedName.UTF8String,
                 d.uniqueID.UTF8String, d.deviceType.UTF8String);
  }
}

static NSArray<AVCaptureDevice *> *DiscoverVideoDevices() {
  // NOTE: Continuity Camera is critical if you want iPhone devices to appear.
  NSArray<AVCaptureDeviceType> *types = @[
    AVCaptureDeviceTypeBuiltInWideAngleCamera,
    AVCaptureDeviceTypeExternalUnknown,
#if defined(AVCaptureDeviceTypeContinuityCamera)
    AVCaptureDeviceTypeContinuityCamera,
#endif
  ];

  AVCaptureDeviceDiscoverySession *discovery =
      [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:types
                                                             mediaType:AVMediaTypeVideo
                                                              position:AVCaptureDevicePositionUnspecified];
  return discovery.devices ?: @[];
}

static AVCaptureDevice *DeviceForIndex(int idx) {
  NSArray<AVCaptureDevice *> *devices = DiscoverVideoDevices();
  DebugPrintDevices(devices);

  if (devices.count == 0)
    return nil;
  if (idx < 0 || idx >= (int)devices.count)
    return nil;
  return devices[(NSUInteger)idx];
}
} // namespace

struct CameraSource::Impl {
  explicit Impl(int idx) : deviceIndex(idx) {}
  ~Impl() {
    alive->store(false, std::memory_order_relaxed);
    stop();
  }

  // Returns true if:
  //  - already running, OR
  //  - start is in progress (permission request and/or async startRunning).
  bool start();

  // Internal: assumes permission already granted.
  bool startAuthorized();

  void stop() noexcept;

  bool readLatest(ImageFrame &out);

  void onSample(CMSampleBufferRef sample);

  static void OnSampleTrampoline(void *ctx, CMSampleBufferRef sample) {
    auto *self = static_cast<Impl *>(ctx);
    if (self)
      self->onSample(sample);
  }

  int deviceIndex = 0;

  AVCaptureSession *session = nil;
  AVCaptureDeviceInput *input = nil;
  AVCaptureVideoDataOutput *output = nil;

  dispatch_queue_t captureQueue = nullptr;
  PDFrameDelegate *delegate = nil;

  std::mutex m;
  bool running = false;
  bool startRequested = false;

  LatestFrame latest;

  std::shared_ptr<std::atomic<bool>> alive = std::make_shared<std::atomic<bool>>(true);

  // simple counter for rate-limited logging
  uint64_t logCounter = 0;
};

bool CameraSource::Impl::start() {
  {
    std::lock_guard<std::mutex> lock(m);
    if (running)
      return true;
    if (startRequested)
      return true;
    startRequested = true;
  }

  @autoreleasepool {
    AVAuthorizationStatus st = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];

    if (st == AVAuthorizationStatusAuthorized) {
      return startAuthorized();
    }

    if (st == AVAuthorizationStatusDenied || st == AVAuthorizationStatusRestricted) {
      std::fprintf(stderr, "[pd] Camera permission denied/restricted.\n");
      std::lock_guard<std::mutex> lock(m);
      running = false;
      startRequested = false;
      return false;
    }

    // NotDetermined: request permission (triggers prompt).
    auto aliveToken = alive;
    auto *self = this;

    std::fprintf(stderr, "[pd] Requesting camera permission...\n");
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                             completionHandler:^(BOOL granted) {
                               if (!aliveToken->load(std::memory_order_relaxed))
                                 return;

                               if (!granted) {
                                 std::fprintf(stderr, "[pd] Camera permission NOT granted.\n");
                                 std::lock_guard<std::mutex> lock(self->m);
                                 self->running = false;
                                 self->startRequested = false;
                                 return;
                               }

                               std::fprintf(stderr, "[pd] Camera permission granted.\n");

                               // Build/start in background to avoid blocking UI.
                               dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                                 @autoreleasepool {
                                   if (!aliveToken->load(std::memory_order_relaxed))
                                     return;
                                   self->startAuthorized();
                                 }
                               });
                             }];

    // Treat "request in flight" as success.
    return true;
  }
}

bool CameraSource::Impl::startAuthorized() {
  {
    std::lock_guard<std::mutex> lock(m);
    if (running && session != nil) {
      startRequested = false;
      return true;
    }
  }

  @autoreleasepool {
    AVCaptureDevice *dev = DeviceForIndex(deviceIndex);
    if (!dev) {
      std::fprintf(stderr, "[pd] No camera device for index %d.\n", deviceIndex);
      std::lock_guard<std::mutex> lock(m);
      running = false;
      startRequested = false;
      return false;
    }

    std::fprintf(stderr, "[pd] Using device index %d: %s (uniqueID=%s)\n", deviceIndex, dev.localizedName.UTF8String,
                 dev.uniqueID.UTF8String);

    NSError *err = nil;

    AVCaptureSession *newSession = [[AVCaptureSession alloc] init];
    [newSession beginConfiguration];
    newSession.sessionPreset = AVCaptureSessionPresetHigh;

    AVCaptureDeviceInput *newInput = [AVCaptureDeviceInput deviceInputWithDevice:dev error:&err];
    if (!newInput || err) {
      std::fprintf(stderr, "[pd] Failed to create device input: %s\n", err.localizedDescription.UTF8String);
      [newSession commitConfiguration];
      std::lock_guard<std::mutex> lock(m);
      running = false;
      startRequested = false;
      return false;
    }

    if ([newSession canAddInput:newInput])
      [newSession addInput:newInput];
    else {
      std::fprintf(stderr, "[pd] Session cannot add input.\n");
      [newSession commitConfiguration];
      std::lock_guard<std::mutex> lock(m);
      running = false;
      startRequested = false;
      return false;
    }

    AVCaptureVideoDataOutput *newOutput = [[AVCaptureVideoDataOutput alloc] init];

    // BGRA + Metal + IOSurface (helps CI/Metal path and avoids weird incompat cases).
    newOutput.videoSettings = @{
      (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA),
      (id)kCVPixelBufferMetalCompatibilityKey : @YES,
      (id)kCVPixelBufferIOSurfacePropertiesKey : @{}
    };
    newOutput.alwaysDiscardsLateVideoFrames = YES;

    dispatch_queue_t newQueue = dispatch_queue_create("pd.CameraSource.captureQueue", DISPATCH_QUEUE_SERIAL);
    dispatch_queue_set_specific(newQueue, kCaptureQueueKey, kCaptureQueueKey, nullptr);

    PDFrameDelegate *newDelegate = [[PDFrameDelegate alloc] init];
    newDelegate.ctx = this;
    newDelegate.onSample = &Impl::OnSampleTrampoline;

    [newOutput setSampleBufferDelegate:newDelegate queue:newQueue];

    if ([newSession canAddOutput:newOutput])
      [newSession addOutput:newOutput];
    else {
      std::fprintf(stderr, "[pd] Session cannot add output.\n");
      [newSession commitConfiguration];

      [newOutput setSampleBufferDelegate:nil queue:nullptr];
      newDelegate.ctx = nullptr;
      newDelegate.onSample = nullptr;

      std::lock_guard<std::mutex> lock(m);
      running = false;
      startRequested = false;
      return false;
    }

    // Ensure connection enabled (usually is, but make it explicit).
    AVCaptureConnection *conn = [newOutput connectionWithMediaType:AVMediaTypeVideo];
    if (conn)
      conn.enabled = YES;

    [newSession commitConfiguration];

    // Commit state
    {
      std::lock_guard<std::mutex> lock(m);
      session = newSession;
      input = newInput;
      output = newOutput;
      captureQueue = newQueue;
      delegate = newDelegate;
      running = true;
      startRequested = false;
    }

    // Start running asynchronously
    auto aliveToken = alive;
    AVCaptureSession *sessToStart = newSession;

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      @autoreleasepool {
        if (!aliveToken->load(std::memory_order_relaxed))
          return;
        std::fprintf(stderr, "[pd] Calling startRunning()...\n");
        [sessToStart startRunning];
        std::fprintf(stderr, "[pd] startRunning() returned. isRunning=%d\n", (int)sessToStart.isRunning);
      }
    });

    return true;
  }
}

void CameraSource::Impl::stop() noexcept {
  AVCaptureSession *oldSession = nil;
  AVCaptureVideoDataOutput *oldOutput = nil;
  PDFrameDelegate *oldDelegate = nil;
  dispatch_queue_t oldQueue = nullptr;

  {
    std::lock_guard<std::mutex> lock(m);

    if (latest.pb) {
      ReleasePB(latest.pb);
      latest.pb = nullptr;
      latest.seq = 0;
    }

    startRequested = false;

    if (!running && session == nil && output == nil && delegate == nil)
      return;
    running = false;

    oldSession = session;
    oldOutput = output;
    oldDelegate = delegate;
    oldQueue = captureQueue;

    delegate = nil;
    input = nil;
    output = nil;
    session = nil;
    captureQueue = nullptr;
  }

  @autoreleasepool {
    if (oldOutput)
      [oldOutput setSampleBufferDelegate:nil queue:nullptr];

    if (oldDelegate) {
      oldDelegate.ctx = nullptr;
      oldDelegate.onSample = nullptr;
    }

    if (oldQueue) {
      const bool onCaptureQueue = (dispatch_get_specific(kCaptureQueueKey) == kCaptureQueueKey);
      if (!onCaptureQueue)
        dispatch_sync(oldQueue, ^{
                      });
    }

    if (oldSession) {
      dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
          [oldSession stopRunning];
        }
      });
    }
  }
}

void CameraSource::Impl::onSample(CMSampleBufferRef sample) {
  CVPixelBufferRef pb = CMSampleBufferGetImageBuffer(sample);
  if (!pb)
    return;

  RetainPB(pb);

  std::lock_guard<std::mutex> lock(m);
  if (!running) {
    ReleasePB(pb);
    return;
  }

  if (latest.pb)
    ReleasePB(latest.pb);
  latest.pb = pb;
  latest.seq += 1;

  // rate-limited log
  if ((++logCounter % 120) == 0) {
    std::fprintf(stderr, "[pd] receiving frames on device %d... seq=%llu\n", deviceIndex,
                 (unsigned long long)latest.seq);
  }
}

bool CameraSource::Impl::readLatest(ImageFrame &out) {
  std::lock_guard<std::mutex> lock(m);
  if (!running || !latest.pb)
    return false;

  out = ImageFrame(latest.seq, latest.pb); // retains safely
  return true;
}

// ---------------- CameraSource public API ----------------

CameraSource::CameraSource(int deviceIndex) : _impl(std::make_unique<Impl>(deviceIndex)) {
  // Auto-start (non-blocking; may be pending permission).
  _impl->start();
}

CameraSource::~CameraSource() { release(); }

CameraSource::CameraSource(CameraSource &&other) noexcept : _impl(std::move(other._impl)) {}

CameraSource &CameraSource::operator=(CameraSource &&other) noexcept {
  if (this == &other)
    return *this;
  release();
  _impl = std::move(other._impl);
  return *this;
}

bool CameraSource::read(ImageFrame &out) {
  if (!_impl)
    return false;
  return _impl->readLatest(out);
}

void CameraSource::release() noexcept {
  if (_impl)
    _impl->stop();
}

} // namespace pd
