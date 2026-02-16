#import "ImageViewer.h"
#import "CameraSource.h" // pd::ImageFrame

#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreImage/CoreImage.h>
#import <QuartzCore/QuartzCore.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

static inline bool PDIsMainThread() { return [NSThread isMainThread]; }

static inline void PDEnsureApp() {
  if (NSApp == nil) {
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp finishLaunching];
  } else if ([NSApp activationPolicy] != NSApplicationActivationPolicyRegular) {
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
  }
}

static inline NSString *PDToNSString(const std::string &s) {
  return [[NSString alloc] initWithBytes:s.data() length:s.size() encoding:NSUTF8StringEncoding];
}

static inline NSSize PDPixelBufferSize(CVPixelBufferRef pb) {
  return NSMakeSize((CGFloat)CVPixelBufferGetWidth(pb), (CGFloat)CVPixelBufferGetHeight(pb));
}

static inline int PDLowerKeyFromEvent(NSEvent *e) {
  NSString *chars = [[e charactersIgnoringModifiers] lowercaseString];
  if (chars.length > 0)
    return (int)[chars characterAtIndex:0];
  return -1;
}

#pragma mark - Window delegate

@interface PDWindowDelegate : NSObject <NSWindowDelegate>
@property(nonatomic, assign) void *ctx;
@property(nonatomic, assign) void (*onWillClose)(void *ctx, NSWindow *w);
@end

@implementation PDWindowDelegate
- (void)windowWillClose:(NSNotification *)n {
  if (self.onWillClose)
    self.onWillClose(self.ctx, (NSWindow *)n.object);
}
@end

#pragma mark - Key-capturing view (reliable)

@interface PDKeyView : NSView
@property(nonatomic, assign) void *ctx;
@property(nonatomic, assign) void (*onKey)(void *ctx, int key);
@end

@implementation PDKeyView
- (BOOL)acceptsFirstResponder {
  return YES;
}
- (BOOL)becomeFirstResponder {
  return YES;
}

- (void)keyDown:(NSEvent *)event {
  int k = PDLowerKeyFromEvent(event);
  if (k >= 0 && self.onKey)
    self.onKey(self.ctx, k);
}
@end

namespace pd {

struct ImageViewer::Impl {
  struct WindowState {
    bool open = false;

    NSWindow *window = nil;
    PDKeyView *hostView = nil; // layer-backed, key-capturing
    CALayer *imageLayer = nil; // set contents to CGImage

    PDWindowDelegate *winDelegate = nil;
    NSSize lastContentSize = NSMakeSize(0, 0);
  };

  mutable std::mutex m;
  std::unordered_map<std::string, WindowState> windows;

  CIContext *ci = nil;
  CGColorSpaceRef colorSpace = nullptr;

  int lastKey = -1;

  Impl();
  ~Impl();

  WindowState &ensureWindowOnMain(const std::string &name, NSSize contentSize);
  void showOnMain(const std::string &name, const ImageFrame &frame);

  static void OnWillCloseTrampoline(void *ctx, NSWindow *w);
  void handleWillClose(NSWindow *w);

  static void OnKeyTrampoline(void *ctx, int key);

  void closeOnMain(const std::string &name);
  void closeAllOnMain();
  bool isOpenLocked(const std::string &name) const;
};

ImageViewer::Impl::Impl() {
  PDEnsureApp();

  // CPU CoreImage context (debug viewer). Stable + simple.
  ci = [CIContext contextWithOptions:nil];
  colorSpace = CGColorSpaceCreateDeviceRGB();
}

ImageViewer::Impl::~Impl() {
  if (PDIsMainThread()) {
    std::lock_guard<std::mutex> lock(m);
    closeAllOnMain();
  } else {
    Impl *self = this;
    dispatch_sync(dispatch_get_main_queue(), ^{
      std::lock_guard<std::mutex> lock(self->m);
      self->closeAllOnMain();
    });
  }

  if (colorSpace) {
    CGColorSpaceRelease(colorSpace);
    colorSpace = nullptr;
  }
}

void ImageViewer::Impl::OnWillCloseTrampoline(void *ctx, NSWindow *w) {
  auto *self = static_cast<Impl *>(ctx);
  if (!self)
    return;
  std::lock_guard<std::mutex> lock(self->m);
  self->handleWillClose(w);
}

void ImageViewer::Impl::OnKeyTrampoline(void *ctx, int key) {
  auto *self = static_cast<Impl *>(ctx);
  if (!self)
    return;
  std::lock_guard<std::mutex> lock(self->m);
  self->lastKey = key;
}

void ImageViewer::Impl::handleWillClose(NSWindow *w) {
  for (auto &kv : windows) {
    WindowState &st = kv.second;
    if (st.window == w) {
      st.open = false;
      st.window = nil;
      st.hostView = nil;
      st.imageLayer = nil;
      st.winDelegate = nil;
      break;
    }
  }
}

ImageViewer::Impl::WindowState &ImageViewer::Impl::ensureWindowOnMain(const std::string &name, NSSize contentSize) {
  auto it = windows.find(name);
  if (it == windows.end())
    it = windows.emplace(name, WindowState{}).first;
  WindowState &st = it->second;

  if (!st.open || st.window == nil) {
    const NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable |
                             NSWindowStyleMaskResizable;

    const CGFloat w = (contentSize.width > 0 ? contentSize.width : 640);
    const CGFloat h = (contentSize.height > 0 ? contentSize.height : 480);

    NSRect r = NSMakeRect(200, 200, w, h);
    st.window = [[NSWindow alloc] initWithContentRect:r styleMask:style backing:NSBackingStoreBuffered defer:NO];
    st.window.title = PDToNSString(name);

    // Key-capable, layer-backed view.
    st.hostView = [[PDKeyView alloc] initWithFrame:NSMakeRect(0, 0, w, h)];
    st.hostView.ctx = this;
    st.hostView.onKey = &Impl::OnKeyTrampoline;
    st.hostView.wantsLayer = YES;
    st.hostView.autoresizingMask = (NSViewWidthSizable | NSViewHeightSizable);

    // Layer that holds the current frame.
    st.imageLayer = [CALayer layer];
    st.imageLayer.frame = st.hostView.bounds;
    st.imageLayer.contentsGravity = kCAGravityResizeAspect;
    st.imageLayer.magnificationFilter = kCAFilterNearest;
    st.imageLayer.minificationFilter = kCAFilterLinear;
    [st.hostView.layer addSublayer:st.imageLayer];

    st.window.contentView = st.hostView;

    st.winDelegate = [[PDWindowDelegate alloc] init];
    st.winDelegate.ctx = this;
    st.winDelegate.onWillClose = &Impl::OnWillCloseTrampoline;
    st.window.delegate = st.winDelegate;

    st.open = true;
    st.lastContentSize = NSMakeSize(w, h);

    [st.window makeKeyAndOrderFront:nil];
    [st.window makeFirstResponder:st.hostView];
    [NSApp activateIgnoringOtherApps:YES];
  }

  // Auto-resize window to frame size (requested behavior).
  if (contentSize.width > 0 && contentSize.height > 0) {
    if (st.lastContentSize.width != contentSize.width || st.lastContentSize.height != contentSize.height) {
      [st.window setContentSize:contentSize];
      st.lastContentSize = contentSize;
    }
  }

  // Keep layer sized to view bounds (covers manual resize too).
  st.imageLayer.frame = st.hostView.bounds;

  return st;
}

void ImageViewer::Impl::showOnMain(const std::string &name, const ImageFrame &frame) {
  CVPixelBufferRef pb = frame.pixelBuffer();
  if (!pb)
    return;

  NSSize sz = PDPixelBufferSize(pb);
  WindowState &st = ensureWindowOnMain(name, sz);

  // Ensure the window can receive key events.
  [st.window makeFirstResponder:st.hostView];

  // Convert CVPixelBuffer -> CIImage -> CGImage (debug path).
  CIImage *ciImg = [CIImage imageWithCVPixelBuffer:pb];
  if (!ciImg)
    return;

  CGRect extent = ciImg.extent;
  if (CGRectIsEmpty(extent))
    return;

  // Use the simple, widely-supported overload.
  CGImageRef cg = [ci createCGImage:ciImg fromRect:extent];
  if (!cg)
    return;

  // CALayer retains the contents; release our reference after setting.
  st.imageLayer.contents = (__bridge id)cg;
  CGImageRelease(cg);

  st.imageLayer.frame = st.hostView.bounds;
}

void ImageViewer::Impl::closeOnMain(const std::string &name) {
  auto it = windows.find(name);
  if (it == windows.end())
    return;

  WindowState &st = it->second;
  if (st.window)
    [st.window close];

  windows.erase(it);
}

void ImageViewer::Impl::closeAllOnMain() {
  for (auto &kv : windows) {
    WindowState &st = kv.second;
    if (st.window)
      [st.window close];
  }
  windows.clear();
}

bool ImageViewer::Impl::isOpenLocked(const std::string &name) const {
  auto it = windows.find(name);
  if (it == windows.end())
    return false;
  return it->second.open && it->second.window != nil;
}

// ---------------- ImageViewer public API ----------------

ImageViewer::ImageViewer() : _impl(std::make_unique<Impl>()) {}
ImageViewer::~ImageViewer() = default;

ImageViewer::ImageViewer(ImageViewer &&other) noexcept : _impl(std::move(other._impl)) {}

ImageViewer &ImageViewer::operator=(ImageViewer &&other) noexcept {
  if (this == &other)
    return *this;
  _impl = std::move(other._impl);
  return *this;
}

void ImageViewer::show(const std::string &windowName, const ImageFrame &frame) {
  if (!_impl)
    return;

  // Call-site should ideally call show() from main thread (your new main loop does).
  // Still safe if called off-main: we dispatch to main.
  ImageFrame local = frame;

  if (PDIsMainThread()) {
    std::lock_guard<std::mutex> lock(_impl->m);
    PDEnsureApp();
    _impl->showOnMain(windowName, local);
  } else {
    Impl *p = _impl.get();
    dispatch_async(dispatch_get_main_queue(), ^{
      std::lock_guard<std::mutex> lock(p->m);
      PDEnsureApp();
      p->showOnMain(windowName, local);
    });
  }
}

int ImageViewer::waitKey(int delayMs) {
  if (!_impl)
    return -1;

  auto pump = [&]() -> int {
    PDEnsureApp();

    NSDate *until = (delayMs <= 0) ? [NSDate dateWithTimeIntervalSinceNow:0]
                                   : [NSDate dateWithTimeIntervalSinceNow:(double)delayMs / 1000.0];

    // Run the run loop so AppKit stays responsive (OpenCV-like explicit pumping).
    while ([until timeIntervalSinceNow] > 0) {
      @autoreleasepool {
        [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:until];
      }
    }

    int k = -1;
    {
      std::lock_guard<std::mutex> lock(_impl->m);
      k = _impl->lastKey;
      _impl->lastKey = -1;
    }
    return k;
  };

  if (PDIsMainThread())
    return pump();

  __block int result = -1;
  dispatch_sync(dispatch_get_main_queue(), ^{
    result = pump();
  });
  return result;
}

void ImageViewer::close(const std::string &windowName) {
  if (!_impl)
    return;

  if (PDIsMainThread()) {
    std::lock_guard<std::mutex> lock(_impl->m);
    _impl->closeOnMain(windowName);
  } else {
    Impl *p = _impl.get();
    dispatch_async(dispatch_get_main_queue(), ^{
      std::lock_guard<std::mutex> lock(p->m);
      p->closeOnMain(windowName);
    });
  }
}

void ImageViewer::closeAll() {
  if (!_impl)
    return;

  if (PDIsMainThread()) {
    std::lock_guard<std::mutex> lock(_impl->m);
    _impl->closeAllOnMain();
  } else {
    Impl *p = _impl.get();
    dispatch_async(dispatch_get_main_queue(), ^{
      std::lock_guard<std::mutex> lock(p->m);
      p->closeAllOnMain();
    });
  }
}

bool ImageViewer::isOpen(const std::string &windowName) const {
  if (!_impl)
    return false;
  std::lock_guard<std::mutex> lock(_impl->m);
  return _impl->isOpenLocked(windowName);
}

} // namespace pd
