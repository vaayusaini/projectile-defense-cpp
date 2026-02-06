#import <AVFoundation/AVFoundation.h>
#import <AVKit/AVKit.h>
#import <AppKit/AppKit.h>
#include <cmath>

// Helper object to:
// 1) Seek to first frame when ready
// 2) Enforce a desired playback rate when user hits Play in AVPlayerView
@interface PlayerHelper : NSObject
@property(nonatomic, strong) AVPlayer *player;
@property(nonatomic, strong) AVPlayerItem *item;
@property(nonatomic, assign) float desiredRate;
@property(nonatomic, assign) BOOL didPrimeFirstFrame;

- (void)startObserving;
- (void)stopObserving;
@end

@implementation PlayerHelper

- (void)startObserving {
    [self.item addObserver:self
                forKeyPath:@"status"
                   options:(NSKeyValueObservingOptionInitial | NSKeyValueObservingOptionNew)
                   context:nil];

    [self.player addObserver:self forKeyPath:@"rate" options:NSKeyValueObservingOptionNew context:nil];
}

- (void)stopObserving {
    @try {
        [self.item removeObserver:self forKeyPath:@"status"];
    } @catch (...) {
    } @
    try {
        [self.player removeObserver:self forKeyPath:@"rate"];
    } @catch (...) {
    }
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)obj change:(NSDictionary *)change context:(void *)ctx {
    if (obj == self.item && [keyPath isEqualToString:@"status"]) {
        if (self.item.status == AVPlayerItemStatusReadyToPlay && !self.didPrimeFirstFrame) {
            self.didPrimeFirstFrame = YES;

            // Seek to start and pause so the first frame displays.
            [self.player seekToTime:kCMTimeZero
                    toleranceBefore:kCMTimeZero
                     toleranceAfter:kCMTimeZero
                  completionHandler:^(__unused BOOL finished) {
                    [self.player pause];
                  }];
        }
        return;
    }

    if (obj == self.player && [keyPath isEqualToString:@"rate"]) {
        float r = self.player.rate;

        // If user started playback (rate > 0), force desiredRate (e.g. 2x).
        if (r > 0.0f && std::fabs(r - self.desiredRate) > 0.001f) {
            self.player.rate = self.desiredRate;
        }
        return;
    }

    [super observeValueForKeyPath:keyPath ofObject:obj change:change context:ctx];
}

@end

namespace videowidget {

static void quitApp() {
    [NSApp terminate:nil];
}

int playVideoThenExit(const char *path) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        if (!path || !path[0])
            return 1;

        NSString *nsPath = [NSString stringWithUTF8String:path];
        if (!nsPath)
            return 1;

        NSURL *url = [NSURL fileURLWithPath:nsPath];
        if (!url)
            return 1;

        AVPlayerItem *item = [AVPlayerItem playerItemWithURL:url];
        if (!item)
            return 1;

        AVPlayer *player = [AVPlayer playerWithPlayerItem:item];
        if (!player)
            return 1;

        // Window
        NSRect screen = [NSScreen mainScreen].visibleFrame;
        CGFloat w = screen.size.width * 0.7;
        CGFloat h = screen.size.height * 0.7;
        NSRect rect = NSMakeRect(NSMidX(screen) - w / 2.0, NSMidY(screen) - h / 2.0, w, h);

        NSWindow *window = [[NSWindow alloc]
            initWithContentRect:rect
                      styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
                        backing:NSBackingStoreBuffered
                          defer:NO];

        [window setTitle:@"Video"];
        [window makeKeyAndOrderFront:nil];

        AVPlayerView *playerView = [[AVPlayerView alloc] initWithFrame:window.contentView.bounds];
        playerView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        playerView.player = player;
        playerView.videoGravity = AVLayerVideoGravityResizeAspect;
        [window setContentView:playerView];

        [NSApp activateIgnoringOtherApps:YES];

        PlayerHelper *helper = [PlayerHelper new];
        helper.player = player;
        helper.item = item;
        helper.desiredRate = 2.0f;
        [helper startObserving];

        [player pause];

        NSNotificationCenter *nc = [NSNotificationCenter defaultCenter];

        id endToken = [nc addObserverForName:AVPlayerItemDidPlayToEndTimeNotification
                                      object:item
                                       queue:[NSOperationQueue mainQueue]
                                  usingBlock:^(__unused NSNotification *note) {
                                    [helper stopObserving];
                                    quitApp();
                                  }];

        id closeToken = [nc addObserverForName:NSWindowWillCloseNotification
                                        object:window
                                         queue:[NSOperationQueue mainQueue]
                                    usingBlock:^(__unused NSNotification *note) {
                                      [helper stopObserving];
                                      quitApp();
                                    }];

        [NSApp run];

        // Usually not reached due to terminate:, but harmless.
        [helper stopObserving];
        [nc removeObserver:endToken];
        [nc removeObserver:closeToken];
        return 0;
    }
}

} // namespace videowidget