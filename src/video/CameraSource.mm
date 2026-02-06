#include "CameraSource.h"
#include <AVFoundation/AVFoundation.h>

namespace pd {
CameraSource::CameraSource(int device) {
    AVCaptureSession *captureSession = [[AVCaptureSession alloc] init];
    [captureSession beginConfiguration];

    AVCaptureDevice *videoDevice = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeContinuityCamera
                                                                      mediaType:AVMediaTypeVideo
                                                                       position:AVCaptureDevicePositionUnspecified];

    if (!videoDevice) {
        [captureSession commitConfiguration];
    }

    NSError *error = nil;
    AVCaptureDeviceInput *videoDeviceInput = [AVCaptureDeviceInput deviceInputWithDevice:videoDevice error:&error];

    if (!videoDeviceInput || ![captureSession canAddInput:videoDeviceInput]) {
        [captureSession commitConfiguration];
    }

    [captureSession addInput:videoDeviceInput];
    [captureSession commitConfiguration];
}

bool CameraSource::read(ImageFrame &out) {
    @autoreleasepool {
    }
}

} // namespace pd