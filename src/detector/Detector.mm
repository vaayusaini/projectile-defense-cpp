#import "Detector.h"

#import "../videostream/CameraSource.h"

#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace pd {

namespace {

constexpr const char *kDetectorShaders = R"metal(
#include <metal_stdlib>
using namespace metal;

struct BGParams {
  float threshold;
  float learningRate;
  float foregroundUpdateScale;
};

kernel void initializeBackground(texture2d<float, access::read> inputTex [[texture(0)]],
                                 texture2d<float, access::write> backgroundTex [[texture(1)]],
                                 texture2d<float, access::write> maskTex [[texture(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= inputTex.get_width() || gid.y >= inputTex.get_height())
    return;

  const float4 rgba = inputTex.read(gid);
  const float luma = dot(rgba.rgb, float3(0.299f, 0.587f, 0.114f));
  backgroundTex.write(float4(luma, 0.0f, 0.0f, 1.0f), gid);
  maskTex.write(float4(0.0f), gid);
}

kernel void subtractBackground(texture2d<float, access::read> inputTex [[texture(0)]],
                               texture2d<float, access::read_write> backgroundTex [[texture(1)]],
                               texture2d<float, access::write> maskTex [[texture(2)]],
                               constant BGParams &params [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= inputTex.get_width() || gid.y >= inputTex.get_height())
    return;

  const float4 rgba = inputTex.read(gid);
  const float luma = dot(rgba.rgb, float3(0.299f, 0.587f, 0.114f));
  const float bg = backgroundTex.read(gid).r;
  const float diff = fabs(luma - bg);

  const float fg = (diff > params.threshold) ? 1.0f : 0.0f;
  const float fgScale = mix(params.foregroundUpdateScale, 1.0f, 1.0f - fg);
  const float lr = params.learningRate * fgScale;
  const float nextBg = mix(bg, luma, lr);

  backgroundTex.write(float4(nextBg, 0.0f, 0.0f, 1.0f), gid);
  maskTex.write(float4(fg, 0.0f, 0.0f, 1.0f), gid);
}

kernel void erode3x3(texture2d<float, access::read> inputMask [[texture(0)]],
                     texture2d<float, access::write> outputMask [[texture(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= inputMask.get_width() || gid.y >= inputMask.get_height())
    return;

  const int width = int(inputMask.get_width());
  const int height = int(inputMask.get_height());
  const int x = int(gid.x);
  const int y = int(gid.y);

  float v = 1.0f;
  for (int dy = -1; dy <= 1; ++dy) {
    const int sy = clamp(y + dy, 0, height - 1);
    for (int dx = -1; dx <= 1; ++dx) {
      const int sx = clamp(x + dx, 0, width - 1);
      v = min(v, inputMask.read(uint2(uint(sx), uint(sy))).r);
    }
  }
  outputMask.write(float4(v, 0.0f, 0.0f, 1.0f), gid);
}

kernel void dilate3x3(texture2d<float, access::read> inputMask [[texture(0)]],
                      texture2d<float, access::write> outputMask [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= inputMask.get_width() || gid.y >= inputMask.get_height())
    return;

  const int width = int(inputMask.get_width());
  const int height = int(inputMask.get_height());
  const int x = int(gid.x);
  const int y = int(gid.y);

  float v = 0.0f;
  for (int dy = -1; dy <= 1; ++dy) {
    const int sy = clamp(y + dy, 0, height - 1);
    for (int dx = -1; dx <= 1; ++dx) {
      const int sx = clamp(x + dx, 0, width - 1);
      v = max(v, inputMask.read(uint2(uint(sx), uint(sy))).r);
    }
  }
  outputMask.write(float4(v, 0.0f, 0.0f, 1.0f), gid);
}
)metal";

struct BGParams {
  float threshold = 0.16f;
  float learningRate = 0.02f;
  float foregroundUpdateScale = 0.1f;
};

struct RawCandidate {
  cv::Point2f center;
  int area = 0;
};

inline int clampInt(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

inline float clampFloat(float v, float lo, float hi) {
  return std::max(lo, std::min(v, hi));
}

inline float distanceSquared(const cv::Point2f &a, const cv::Point2f &b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return dx * dx + dy * dy;
}

inline MTLSize makeThreadgroupSize() {
  return MTLSizeMake(16, 16, 1);
}

inline MTLSize makeThreadgroupCount(int width, int height, MTLSize tgSize) {
  const NSUInteger groupsX = (static_cast<NSUInteger>(width) + tgSize.width - 1) / tgSize.width;
  const NSUInteger groupsY = (static_cast<NSUInteger>(height) + tgSize.height - 1) / tgSize.height;
  return MTLSizeMake(groupsX, groupsY, 1);
}

void drawPointIntoBGRA(uint8_t *base, int width, int height, int stride, const Pixel &point, int radius) {
  if (!base || width <= 0 || height <= 0 || stride <= 0)
    return;

  const int r = std::max(1, radius);
  const int r2 = r * r;

  const int minY = clampInt(point.y - r, 0, height - 1);
  const int maxY = clampInt(point.y + r, 0, height - 1);
  const int minX = clampInt(point.x - r, 0, width - 1);
  const int maxX = clampInt(point.x + r, 0, width - 1);

  for (int y = minY; y <= maxY; ++y) {
    uint8_t *row = base + y * stride;
    for (int x = minX; x <= maxX; ++x) {
      const int dx = x - point.x;
      const int dy = y - point.y;
      if (dx * dx + dy * dy > r2)
        continue;

      uint8_t *px = row + x * 4;
      px[0] = 0;   // B
      px[1] = 0;   // G
      px[2] = 255; // R
      px[3] = 255; // A
    }
  }
}

class Track {
public:
  Track(int trackId, const cv::Point2f &start, float smoothing) : id(trackId), smoothed(start) {
    kf.init(4, 2, 0, CV_32F);
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                           1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    kf.measurementMatrix =
        (cv::Mat_<float>(2, 4) << 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-2f;
    kf.processNoiseCov.at<float>(2, 2) = 5e-2f;
    kf.processNoiseCov.at<float>(3, 3) = 5e-2f;
    kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 9.0f;
    kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F) * 25.0f;

    kf.statePost.at<float>(0) = start.x;
    kf.statePost.at<float>(1) = start.y;
    kf.statePost.at<float>(2) = 0.0f;
    kf.statePost.at<float>(3) = 0.0f;

    smoothAlpha = clampFloat(smoothing, 0.0f, 1.0f);
  }

  cv::Point2f predict(float gravityY) {
    cv::Mat prediction = kf.predict();
    prediction.at<float>(1) += 0.5f * gravityY;
    prediction.at<float>(3) += gravityY;
    kf.statePre.at<float>(1) = prediction.at<float>(1);
    kf.statePre.at<float>(3) = prediction.at<float>(3);
    predicted = cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
    return predicted;
  }

  void correct(const cv::Point2f &measurement) {
    cv::Mat z(2, 1, CV_32F);
    z.at<float>(0) = measurement.x;
    z.at<float>(1) = measurement.y;
    cv::Mat corrected = kf.correct(z);
    const cv::Point2f correctedPt(corrected.at<float>(0), corrected.at<float>(1));
    smoothed = smoothAlpha * correctedPt + (1.0f - smoothAlpha) * smoothed;
  }

  Pixel pixel() const {
    Pixel p;
    p.x = static_cast<int>(std::lround(smoothed.x));
    p.y = static_cast<int>(std::lround(smoothed.y));
    return p;
  }

  int id = 0;
  int hits = 1;
  int misses = 0;
  cv::Point2f predicted = {};
  cv::Point2f smoothed = {};
  float smoothAlpha = 0.65f;
  cv::KalmanFilter kf;
};

} // namespace

struct Detector::Impl {
  explicit Impl(const DetectorConfig &cfg) : config(cfg) {
    config.foregroundThreshold = clampFloat(config.foregroundThreshold, 0.0f, 1.0f);
    config.backgroundLearningRate = clampFloat(config.backgroundLearningRate, 0.0f, 1.0f);
    config.foregroundUpdateScale = clampFloat(config.foregroundUpdateScale, 0.0f, 1.0f);

    config.morphologyOpenIterations = std::max(0, config.morphologyOpenIterations);
    config.morphologyCloseIterations = std::max(0, config.morphologyCloseIterations);
    config.warmupFrames = std::max(0, config.warmupFrames);

    config.minBlobArea = std::max(1, config.minBlobArea);
    config.maxBlobArea = std::max(config.minBlobArea, config.maxBlobArea);
    config.maxBlobAspectRatio = std::max(1.0f, config.maxBlobAspectRatio);
    config.minBlobFillRatio = clampFloat(config.minBlobFillRatio, 0.0f, 1.0f);
    config.borderIgnorePixels = std::max(0, config.borderIgnorePixels);
    config.maxRawDetections = std::max(1, config.maxRawDetections);

    config.maxAssociationDistancePx = std::max(1.0f, config.maxAssociationDistancePx);
    config.minConfirmedHits = std::max(1, config.minConfirmedHits);
    config.maxMissedFrames = std::max(0, config.maxMissedFrames);
    config.smoothingAlpha = clampFloat(config.smoothingAlpha, 0.0f, 1.0f);

    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::fprintf(stderr, "[pd::Detector] Metal device unavailable.\n");
      return;
    }

    queue = [device newCommandQueue];
    if (!queue) {
      std::fprintf(stderr, "[pd::Detector] Failed to create Metal command queue.\n");
      return;
    }

    NSError *error = nil;
    NSString *src = [NSString stringWithUTF8String:kDetectorShaders];
    library = [device newLibraryWithSource:src options:nil error:&error];
    if (!library) {
      std::fprintf(stderr, "[pd::Detector] Failed to compile Metal shaders: %s\n",
                   error.localizedDescription.UTF8String);
      return;
    }

    initPSO = createPSO(@"initializeBackground");
    bgSubPSO = createPSO(@"subtractBackground");
    erodePSO = createPSO(@"erode3x3");
    dilatePSO = createPSO(@"dilate3x3");
    if (!initPSO || !bgSubPSO || !erodePSO || !dilatePSO)
      return;

    const CVReturn cacheResult = CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr, device, nullptr, &textureCache);
    if (cacheResult != kCVReturnSuccess) {
      textureCache = nullptr;
      std::fprintf(stderr, "[pd::Detector] Failed to create CVMetalTextureCache (%d).\n", int(cacheResult));
    }
  }

  ~Impl() {
    if (textureCache) {
      CVMetalTextureCacheFlush(textureCache, 0);
      CFRelease(textureCache);
      textureCache = nullptr;
    }
  }

  std::vector<Pixel> process(const ImageFrame &frame) {
    CVPixelBufferRef pb = frame.pixelBuffer();
    if (!pb || !device || !queue || !textureCache || !initPSO || !bgSubPSO || !erodePSO || !dilatePSO)
      return {};
    if (CVPixelBufferGetPixelFormatType(pb) != kCVPixelFormatType_32BGRA)
      return {};

    frameWidth = static_cast<int>(CVPixelBufferGetWidth(pb));
    frameHeight = static_cast<int>(CVPixelBufferGetHeight(pb));
    if (frameWidth <= 0 || frameHeight <= 0)
      return {};

    CVMetalTextureRef cvInputTex = nullptr;
    const CVReturn cvResult = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, textureCache, pb, nullptr, MTLPixelFormatBGRA8Unorm, frameWidth, frameHeight, 0, &cvInputTex);
    if (cvResult != kCVReturnSuccess || !cvInputTex)
      return {};

    id<MTLTexture> inputTexture = CVMetalTextureGetTexture(cvInputTex);
    if (!inputTexture) {
      CFRelease(cvInputTex);
      return {};
    }

    if (!ensureWorkingTextures(frameWidth, frameHeight)) {
      CFRelease(cvInputTex);
      return {};
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
      CFRelease(cvInputTex);
      return {};
    }

    const MTLSize tgSize = makeThreadgroupSize();
    const MTLSize tgCount = makeThreadgroupCount(frameWidth, frameHeight, tgSize);

    if (!backgroundInitialized) {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:initPSO];
      [enc setTexture:inputTexture atIndex:0];
      [enc setTexture:backgroundTexture atIndex:1];
      [enc setTexture:maskTexture atIndex:2];
      [enc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
      [enc endEncoding];
      backgroundInitialized = true;
      warmupCounter = 0;
    } else {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:bgSubPSO];
      [enc setTexture:inputTexture atIndex:0];
      [enc setTexture:backgroundTexture atIndex:1];
      [enc setTexture:maskTexture atIndex:2];

      BGParams params;
      params.threshold = config.foregroundThreshold;
      params.learningRate = config.backgroundLearningRate;
      params.foregroundUpdateScale = config.foregroundUpdateScale;
      [enc setBytes:&params length:sizeof(params) atIndex:0];
      [enc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
      [enc endEncoding];
    }

    for (int i = 0; i < config.morphologyOpenIterations; ++i) {
      runMorphStep(cmd, erodePSO, maskTexture, tempMaskTexture, tgCount, tgSize);
      runMorphStep(cmd, dilatePSO, tempMaskTexture, maskTexture, tgCount, tgSize);
    }

    for (int i = 0; i < config.morphologyCloseIterations; ++i) {
      runMorphStep(cmd, dilatePSO, maskTexture, tempMaskTexture, tgCount, tgSize);
      runMorphStep(cmd, erodePSO, tempMaskTexture, maskTexture, tgCount, tgSize);
    }

    [cmd commit];
    [cmd waitUntilCompleted];
    CFRelease(cvInputTex);

    if (cmd.status != MTLCommandBufferStatusCompleted)
      return {};

    if (warmupCounter < config.warmupFrames) {
      ++warmupCounter;
      return advanceTracks({});
    }

    const size_t required = static_cast<size_t>(frameWidth) * static_cast<size_t>(frameHeight);
    cpuMask.resize(required);

    const MTLRegion region =
        MTLRegionMake2D(0, 0, static_cast<NSUInteger>(frameWidth), static_cast<NSUInteger>(frameHeight));
    [maskTexture getBytes:cpuMask.data() bytesPerRow:static_cast<NSUInteger>(frameWidth) fromRegion:region mipmapLevel:0];

    const std::vector<RawCandidate> candidates = extractRawCandidates();
    return advanceTracks(candidates);
  }

private:
  id<MTLComputePipelineState> createPSO(NSString *name) {
    if (!library)
      return nil;

    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) {
      std::fprintf(stderr, "[pd::Detector] Missing Metal function: %s\n", name.UTF8String);
      return nil;
    }

    NSError *error = nil;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) {
      std::fprintf(stderr, "[pd::Detector] Failed to create PSO %s: %s\n", name.UTF8String,
                   error.localizedDescription.UTF8String);
    }
    return pso;
  }

  void runMorphStep(id<MTLCommandBuffer> cmd,
                    id<MTLComputePipelineState> pso,
                    id<MTLTexture> inTex,
                    id<MTLTexture> outTex,
                    MTLSize tgCount,
                    MTLSize tgSize) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setTexture:inTex atIndex:0];
    [enc setTexture:outTex atIndex:1];
    [enc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
    [enc endEncoding];
  }

  bool ensureWorkingTextures(int width, int height) {
    if (width == texWidth && height == texHeight && backgroundTexture && maskTexture && tempMaskTexture)
      return true;

    texWidth = width;
    texHeight = height;
    backgroundInitialized = false;
    tracks.clear();
    nextTrackId = 1;

    MTLTextureDescriptor *bgDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:static_cast<NSUInteger>(width)
                                                          height:static_cast<NSUInteger>(height)
                                                       mipmapped:NO];
    bgDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    bgDesc.storageMode = MTLStorageModeShared;

    MTLTextureDescriptor *maskDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                           width:static_cast<NSUInteger>(width)
                                                          height:static_cast<NSUInteger>(height)
                                                       mipmapped:NO];
    maskDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    maskDesc.storageMode = MTLStorageModeShared;

    backgroundTexture = [device newTextureWithDescriptor:bgDesc];
    maskTexture = [device newTextureWithDescriptor:maskDesc];
    tempMaskTexture = [device newTextureWithDescriptor:maskDesc];

    if (!backgroundTexture || !maskTexture || !tempMaskTexture) {
      std::fprintf(stderr, "[pd::Detector] Failed to allocate textures (%dx%d).\n", width, height);
      return false;
    }

    return true;
  }

  std::vector<RawCandidate> extractRawCandidates() {
    std::vector<RawCandidate> out;
    if (cpuMask.empty() || frameWidth <= 0 || frameHeight <= 0)
      return out;

    const size_t total = static_cast<size_t>(frameWidth) * static_cast<size_t>(frameHeight);
    visited.assign(total, 0);
    queueScratch.clear();
    queueScratch.reserve(2048);
    out.reserve(32);

    constexpr int kNeighbors[8][2] = {
      {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1},
    };

    for (int y = 0; y < frameHeight; ++y) {
      for (int x = 0; x < frameWidth; ++x) {
        const int start = y * frameWidth + x;
        if (visited[start] || cpuMask[start] == 0)
          continue;

        visited[start] = 1;
        queueScratch.clear();
        queueScratch.push_back(start);

        size_t head = 0;
        int area = 0;
        int minX = x, maxX = x, minY = y, maxY = y;
        long long sumX = 0;
        long long sumY = 0;

        while (head < queueScratch.size()) {
          const int idx = queueScratch[head++];
          const int cx = idx % frameWidth;
          const int cy = idx / frameWidth;

          ++area;
          sumX += cx;
          sumY += cy;
          minX = std::min(minX, cx);
          maxX = std::max(maxX, cx);
          minY = std::min(minY, cy);
          maxY = std::max(maxY, cy);

          for (const auto &n : kNeighbors) {
            const int nx = cx + n[0];
            const int ny = cy + n[1];
            if (nx < 0 || ny < 0 || nx >= frameWidth || ny >= frameHeight)
              continue;

            const int nidx = ny * frameWidth + nx;
            if (visited[nidx] || cpuMask[nidx] == 0)
              continue;

            visited[nidx] = 1;
            queueScratch.push_back(nidx);
          }
        }

        if (area < config.minBlobArea || area > config.maxBlobArea)
          continue;

        if (minX <= config.borderIgnorePixels || minY <= config.borderIgnorePixels ||
            maxX >= frameWidth - 1 - config.borderIgnorePixels || maxY >= frameHeight - 1 - config.borderIgnorePixels)
          continue;

        const int boxW = maxX - minX + 1;
        const int boxH = maxY - minY + 1;
        if (boxW <= 0 || boxH <= 0)
          continue;

        const float aspect = static_cast<float>(std::max(boxW, boxH)) / static_cast<float>(std::min(boxW, boxH));
        if (aspect > config.maxBlobAspectRatio)
          continue;

        const float fill = static_cast<float>(area) / static_cast<float>(boxW * boxH);
        if (fill < config.minBlobFillRatio)
          continue;

        RawCandidate c;
        c.center.x = static_cast<float>(sumX) / static_cast<float>(area);
        c.center.y = static_cast<float>(sumY) / static_cast<float>(area);
        c.area = area;
        out.push_back(c);
      }
    }

    std::sort(out.begin(), out.end(), [](const RawCandidate &a, const RawCandidate &b) { return a.area > b.area; });
    if (static_cast<int>(out.size()) > config.maxRawDetections)
      out.resize(config.maxRawDetections);
    return out;
  }

  std::vector<Pixel> advanceTracks(const std::vector<RawCandidate> &candidates) {
    const float maxDistSq = config.maxAssociationDistancePx * config.maxAssociationDistancePx;

    for (Track &t : tracks)
      t.predict(config.ballisticGravityY);

    std::vector<bool> used(candidates.size(), false);

    for (Track &t : tracks) {
      int best = -1;
      float bestD2 = std::numeric_limits<float>::max();
      for (size_t i = 0; i < candidates.size(); ++i) {
        if (used[i])
          continue;
        const float d2 = distanceSquared(t.predicted, candidates[i].center);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = static_cast<int>(i);
        }
      }

      if (best >= 0 && bestD2 <= maxDistSq) {
        t.correct(candidates[best].center);
        t.hits += 1;
        t.misses = 0;
        used[best] = true;
      } else {
        t.misses += 1;
      }
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
      if (used[i])
        continue;
      tracks.emplace_back(nextTrackId++, candidates[i].center, config.smoothingAlpha);
    }

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                [&](const Track &t) { return t.misses > config.maxMissedFrames; }),
                 tracks.end());

    std::vector<Pixel> visible;
    visible.reserve(tracks.size());
    for (const Track &t : tracks) {
      if (t.hits < config.minConfirmedHits || t.misses != 0)
        continue;
      Pixel p = t.pixel();
      p.x = clampInt(p.x, 0, frameWidth - 1);
      p.y = clampInt(p.y, 0, frameHeight - 1);
      visible.push_back(p);
    }
    return visible;
  }

  DetectorConfig config;

  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> initPSO = nil;
  id<MTLComputePipelineState> bgSubPSO = nil;
  id<MTLComputePipelineState> erodePSO = nil;
  id<MTLComputePipelineState> dilatePSO = nil;

  CVMetalTextureCacheRef textureCache = nullptr;

  id<MTLTexture> backgroundTexture = nil;
  id<MTLTexture> maskTexture = nil;
  id<MTLTexture> tempMaskTexture = nil;

  int texWidth = 0;
  int texHeight = 0;
  int frameWidth = 0;
  int frameHeight = 0;
  bool backgroundInitialized = false;
  int warmupCounter = 0;

  std::vector<uint8_t> cpuMask;
  std::vector<uint8_t> visited;
  std::vector<int> queueScratch;

  int nextTrackId = 1;
  std::vector<Track> tracks;
};

Detector::Detector(const DetectorConfig &config) : _impl(std::make_unique<Impl>(config)) {}
Detector::~Detector() = default;
Detector::Detector(Detector &&other) noexcept = default;
Detector &Detector::operator=(Detector &&other) noexcept = default;

std::vector<Pixel> Detector::process(const ImageFrame &frame) {
  if (!_impl)
    return {};
  return _impl->process(frame);
}

void drawPoint(ImageFrame &frame, const Pixel &point, int radius) {
  CVPixelBufferRef pb = frame.pixelBuffer();
  if (!pb)
    return;

  if (CVPixelBufferGetPixelFormatType(pb) != kCVPixelFormatType_32BGRA)
    return;

  CVPixelBufferLockBaseAddress(pb, 0);
  void *base = CVPixelBufferGetBaseAddress(pb);
  if (!base) {
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return;
  }

  const int width = static_cast<int>(CVPixelBufferGetWidth(pb));
  const int height = static_cast<int>(CVPixelBufferGetHeight(pb));
  const int stride = static_cast<int>(CVPixelBufferGetBytesPerRow(pb));
  drawPointIntoBGRA(static_cast<uint8_t *>(base), width, height, stride, point, radius);

  CVPixelBufferUnlockBaseAddress(pb, 0);
}

void drawPoints(ImageFrame &frame, const std::vector<Pixel> &points, int radius) {
  if (points.empty())
    return;

  CVPixelBufferRef pb = frame.pixelBuffer();
  if (!pb)
    return;

  if (CVPixelBufferGetPixelFormatType(pb) != kCVPixelFormatType_32BGRA)
    return;

  CVPixelBufferLockBaseAddress(pb, 0);
  void *base = CVPixelBufferGetBaseAddress(pb);
  if (!base) {
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return;
  }

  const int width = static_cast<int>(CVPixelBufferGetWidth(pb));
  const int height = static_cast<int>(CVPixelBufferGetHeight(pb));
  const int stride = static_cast<int>(CVPixelBufferGetBytesPerRow(pb));

  auto *ptr = static_cast<uint8_t *>(base);
  for (const Pixel &p : points)
    drawPointIntoBGRA(ptr, width, height, stride, p, radius);

  CVPixelBufferUnlockBaseAddress(pb, 0);
}

} // namespace pd

