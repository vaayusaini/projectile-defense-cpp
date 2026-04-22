#import "Detector.h"

#import "../videostream/CameraSource.h"

#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace pd {

namespace {

constexpr const char *kDetectorShaders = R"metal(
#include <metal_stdlib>
using namespace metal;

struct MOG2Params {
  int frameWidth;
  int frameHeight;
  int scaledWidth;
  int scaledHeight;
  int nMixtures;
  float alphaT;
  float alpha1;
  float prune;
  float Tb;
  float TB;
  float Tg;
  float varInit;
  float varMin;
  float varMax;
  float tau;
  uint detectShadows;
};

struct MorphParams {
  int radius;
};

inline void swapMode(device float4 *means,
                     device float *weights,
                     device float *variances,
                     int idxA,
                     int idxB) {
  const float4 m = means[idxA];
  means[idxA] = means[idxB];
  means[idxB] = m;
  const float w = weights[idxA];
  weights[idxA] = weights[idxB];
  weights[idxB] = w;
  const float v = variances[idxA];
  variances[idxA] = variances[idxB];
  variances[idxB] = v;
}

inline bool detectShadowGMM(float3 data,
                            int nmodes,
                            int base,
                            device const float4 *means,
                            device const float *weights,
                            device const float *variances,
                            float Tb,
                            float TB,
                            float tau) {
  float tWeight = 0.0f;
  for (int mode = 0; mode < nmodes; ++mode) {
    const float3 mean = means[base + mode].xyz;
    const float numerator = dot(data, mean);
    const float denominator = dot(mean, mean);

    if (denominator > 1e-6f && numerator <= denominator && numerator >= tau * denominator) {
      const float a = numerator / denominator;
      const float3 d = a * mean - data;
      const float dist2a = dot(d, d);
      if (dist2a < Tb * variances[base + mode] * a * a)
        return true;
    }

    tWeight += weights[base + mode];
    if (tWeight > TB)
      return false;
  }
  return false;
}

kernel void mog2Update(texture2d<float, access::read> inputTex [[texture(0)]],
                       texture2d<float, access::write> maskTex [[texture(1)]],
                       device float4 *means [[buffer(0)]],
                       device float *weights [[buffer(1)]],
                       device float *variances [[buffer(2)]],
                       device uchar *modesUsed [[buffer(3)]],
                       constant MOG2Params &params [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= uint(params.scaledWidth) || gid.y >= uint(params.scaledHeight))
    return;

  const float inW = float(params.frameWidth);
  const float inH = float(params.frameHeight);
  const float outW = float(params.scaledWidth);
  const float outH = float(params.scaledHeight);
  const int sx = clamp(int(((float(gid.x) + 0.5f) * inW) / outW), 0, params.frameWidth - 1);
  const int sy = clamp(int(((float(gid.y) + 0.5f) * inH) / outH), 0, params.frameHeight - 1);
  const float3 data = inputTex.read(uint2(uint(sx), uint(sy))).rgb * 255.0f;

  const int pixelIndex = int(gid.y) * params.scaledWidth + int(gid.x);
  const int base = pixelIndex * params.nMixtures;
  int nmodes = clamp(int(modesUsed[pixelIndex]), 0, params.nMixtures);

  bool background = false;
  bool fitsPDF = false;
  float totalWeight = 0.0f;

  for (int mode = 0; mode < nmodes; ++mode) {
    const int idx = base + mode;
    float weight = params.alpha1 * weights[idx] + params.prune;
    int swapCount = 0;

    if (!fitsPDF) {
      const float3 mean = means[idx].xyz;
      float variance = variances[idx];
      const float3 dData = mean - data;
      const float dist2 = dot(dData, dData);

      if (totalWeight < params.TB && dist2 < params.Tb * variance)
        background = true;

      if (dist2 < params.Tg * variance) {
        fitsPDF = true;
        weight += params.alphaT;

        const float k = params.alphaT / weight;
        const float3 nextMean = mean - k * dData;
        variance += k * (dist2 - variance);
        variance = clamp(variance, params.varMin, params.varMax);

        means[idx] = float4(nextMean, 0.0f);
        variances[idx] = variance;

        for (int i = mode; i > 0; --i) {
          const int prevIdx = base + i - 1;
          if (weight < weights[prevIdx])
            break;
          swapMode(means, weights, variances, base + i, prevIdx);
          ++swapCount;
        }
      }
    }

    if (weight < -params.prune) {
      weight = 0.0f;
      nmodes--;
    }

    weights[base + mode - swapCount] = weight;
    totalWeight += weight;
  }

  const float invWeight = (fabs(totalWeight) > 1e-6f) ? (1.0f / totalWeight) : 0.0f;
  for (int mode = 0; mode < nmodes; ++mode)
    weights[base + mode] *= invWeight;

  if (!fitsPDF && params.alphaT > 0.0f) {
    int mode = (nmodes == params.nMixtures) ? (params.nMixtures - 1) : nmodes++;
    const int idx = base + mode;
    if (nmodes == 1) {
      weights[idx] = 1.0f;
    } else {
      weights[idx] = params.alphaT;
      for (int i = 0; i < (nmodes - 1); ++i)
        weights[base + i] *= params.alpha1;
    }
    means[idx] = float4(data, 0.0f);
    variances[idx] = params.varInit;

    for (int i = nmodes - 1; i > 0; --i) {
      if (params.alphaT < weights[base + i - 1])
        break;
      swapMode(means, weights, variances, base + i, base + i - 1);
    }
  }

  modesUsed[pixelIndex] = uchar(clamp(nmodes, 0, params.nMixtures));

  float foreground = background ? 0.0f : 1.0f;
  if (foreground > 0.5f && params.detectShadows != 0 &&
      detectShadowGMM(data, nmodes, base, means, weights, variances, params.Tb, params.TB, params.tau)) {
    foreground = 0.0f;
  }

  maskTex.write(float4(foreground, 0.0f, 0.0f, 1.0f), gid);
}

kernel void dilateNxN(texture2d<float, access::read> inputMask [[texture(0)]],
                      texture2d<float, access::write> outputMask [[texture(1)]],
                      constant MorphParams &params [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= inputMask.get_width() || gid.y >= inputMask.get_height())
    return;

  const int width = int(inputMask.get_width());
  const int height = int(inputMask.get_height());
  const int x = int(gid.x);
  const int y = int(gid.y);
  const int r = max(0, params.radius);

  float v = 0.0f;
  for (int dy = -r; dy <= r; ++dy) {
    const int sy = clamp(y + dy, 0, height - 1);
    for (int dx = -r; dx <= r; ++dx) {
      const int sx = clamp(x + dx, 0, width - 1);
      v = max(v, inputMask.read(uint2(uint(sx), uint(sy))).r);
    }
  }
  outputMask.write(float4(v, 0.0f, 0.0f, 1.0f), gid);
}

kernel void erodeNxN(texture2d<float, access::read> inputMask [[texture(0)]],
                     texture2d<float, access::write> outputMask [[texture(1)]],
                     constant MorphParams &params [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= inputMask.get_width() || gid.y >= inputMask.get_height())
    return;

  const int width = int(inputMask.get_width());
  const int height = int(inputMask.get_height());
  const int x = int(gid.x);
  const int y = int(gid.y);
  const int r = max(0, params.radius);

  float v = 1.0f;
  for (int dy = -r; dy <= r; ++dy) {
    const int sy = clamp(y + dy, 0, height - 1);
    for (int dx = -r; dx <= r; ++dx) {
      const int sx = clamp(x + dx, 0, width - 1);
      v = min(v, inputMask.read(uint2(uint(sx), uint(sy))).r);
    }
  }
  outputMask.write(float4(v, 0.0f, 0.0f, 1.0f), gid);
}
)metal";

struct MOG2Params {
  int frameWidth = 0;
  int frameHeight = 0;
  int scaledWidth = 0;
  int scaledHeight = 0;
  int nMixtures = 5;
  float alphaT = 0.0f;
  float alpha1 = 1.0f;
  float prune = 0.0f;
  float Tb = 16.0f;
  float TB = 0.7f;
  float Tg = 9.0f;
  float varInit = 15.0f;
  float varMin = 4.0f;
  float varMax = 75.0f;
  float tau = 0.5f;
  uint32_t detectShadows = 0;
};

struct MorphParams {
  int radius = 2;
};

constexpr int kMog2Mixtures = 5;
constexpr float kMog2VarThresholdGen = 9.0f;
constexpr float kMog2VarInit = 15.0f;
constexpr float kMog2VarMin = 4.0f;
constexpr float kMog2VarMax = 75.0f;
constexpr float kMog2ComplexityReduction = 0.05f;
constexpr float kMog2ShadowTau = 0.5f;

inline int clampInt(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

inline float clampFloat(float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); }

inline MTLSize makeThreadgroupSize() { return MTLSizeMake(16, 16, 1); }

inline MTLSize makeThreadgroupCount(int width, int height, MTLSize tgSize) {
  const NSUInteger gx = (static_cast<NSUInteger>(width) + tgSize.width - 1) / tgSize.width;
  const NSUInteger gy = (static_cast<NSUInteger>(height) + tgSize.height - 1) / tgSize.height;
  return MTLSizeMake(gx, gy, 1);
}

struct RectI {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

inline RectI clampRectToFrame(const RectI &r, int width, int height) {
  const int x0 = clampInt(r.x, 0, width);
  const int y0 = clampInt(r.y, 0, height);
  const int x1 = clampInt(r.x + r.width, 0, width);
  const int y1 = clampInt(r.y + r.height, 0, height);
  return RectI{x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)};
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

void drawRectIntoBGRA(uint8_t *base, int width, int height, int stride, RectI rect, int thickness) {
  if (!base || width <= 0 || height <= 0 || stride <= 0)
    return;
  rect = clampRectToFrame(rect, width, height);
  if (rect.width <= 0 || rect.height <= 0)
    return;

  const int t = std::max(1, thickness);
  const int x0 = rect.x;
  const int y0 = rect.y;
  const int x1 = rect.x + rect.width - 1;
  const int y1 = rect.y + rect.height - 1;

  auto drawHorizontal = [&](int y) {
    if (y < 0 || y >= height)
      return;
    uint8_t *row = base + y * stride;
    const int start = clampInt(x0, 0, width - 1);
    const int end = clampInt(x1, 0, width - 1);
    for (int x = start; x <= end; ++x) {
      uint8_t *px = row + x * 4;
      px[0] = 0;
      px[1] = 255; // green
      px[2] = 0;
      px[3] = 255;
    }
  };

  auto drawVertical = [&](int x) {
    if (x < 0 || x >= width)
      return;
    const int start = clampInt(y0, 0, height - 1);
    const int end = clampInt(y1, 0, height - 1);
    for (int y = start; y <= end; ++y) {
      uint8_t *row = base + y * stride;
      uint8_t *px = row + x * 4;
      px[0] = 0;
      px[1] = 255;
      px[2] = 0;
      px[3] = 255;
    }
  };

  for (int i = 0; i < t; ++i) {
    drawHorizontal(y0 + i);
    drawHorizontal(y1 - i);
    drawVertical(x0 + i);
    drawVertical(x1 - i);
  }
}

struct LockedFrameView {
  explicit LockedFrameView(CVPixelBufferRef pb, CVPixelBufferLockFlags flags) : _pb(pb), _flags(flags) {
    if (!_pb)
      return;
    if (CVPixelBufferGetPixelFormatType(_pb) != kCVPixelFormatType_32BGRA)
      return;

    CVPixelBufferLockBaseAddress(_pb, _flags);
    _locked = true;

    void *base = CVPixelBufferGetBaseAddress(_pb);
    if (!base)
      return;

    _width = static_cast<int>(CVPixelBufferGetWidth(_pb));
    _height = static_cast<int>(CVPixelBufferGetHeight(_pb));
    _stride = static_cast<int>(CVPixelBufferGetBytesPerRow(_pb));
    if (_width <= 0 || _height <= 0 || _stride < _width * 4)
      return;

    _base = static_cast<uint8_t *>(base);
    _valid = true;
  }

  ~LockedFrameView() {
    if (_locked)
      CVPixelBufferUnlockBaseAddress(_pb, _flags);
  }

  LockedFrameView(const LockedFrameView &) = delete;
  LockedFrameView &operator=(const LockedFrameView &) = delete;

  bool valid() const { return _valid; }
  uint8_t *base() const { return _base; }
  int width() const { return _width; }
  int height() const { return _height; }
  int stride() const { return _stride; }

private:
  CVPixelBufferRef _pb = nullptr;
  CVPixelBufferLockFlags _flags = 0;
  bool _locked = false;
  bool _valid = false;
  uint8_t *_base = nullptr;
  int _width = 0;
  int _height = 0;
  int _stride = 0;
};

} // namespace

struct Detector::Impl {
  explicit Impl(const DetectorConfig &cfg) { initialize(cfg); }

  ~Impl() {
    if (textureCache) {
      CVMetalTextureCacheFlush(textureCache, 0);
      CFRelease(textureCache);
      textureCache = nullptr;
    }
  }

  void initialize(const DetectorConfig &cfg) {
    applyConfig(cfg);

    device = MTLCreateSystemDefaultDevice();
    if (!device)
      return;

    queue = [device newCommandQueue];
    if (!queue)
      return;

    NSError *error = nil;
    NSString *src = [NSString stringWithUTF8String:kDetectorShaders];
    library = [device newLibraryWithSource:src options:nil error:&error];
    if (!library)
      return;

    mog2PSO = createPSO(@"mog2Update");
    dilatePSO = createPSO(@"dilateNxN");
    erodePSO = createPSO(@"erodeNxN");

    if (!mog2PSO || !dilatePSO || !erodePSO)
      return;

    const CVReturn cacheResult =
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr, device, nullptr, &textureCache);
    if (cacheResult != kCVReturnSuccess)
      textureCache = nullptr;
  }

  void applyConfig(const DetectorConfig &cfg) {
    config = cfg;
    if (config.imscale <= 0.0)
      config.imscale = 1.0;
    config.bgHistory = std::max(1, config.bgHistory);
    config.varThreshold = std::max(0.0, config.varThreshold);
    config.bgRatio = clampFloat(static_cast<float>(config.bgRatio), 0.0f, 1.0f);
    config.closeKernelSize = std::max(1, config.closeKernelSize);
    if ((config.closeKernelSize % 2) == 0)
      config.closeKernelSize += 1;
    config.closeIterations = std::max(0, config.closeIterations);
    config.connectivity = (config.connectivity == 4 ? 4 : 8);
    config.minArea = std::max(1, config.minArea);
    config.maxProjectiles = std::max(1, config.maxProjectiles);
    config.maxAspect = std::max(config.maxAspect, config.minAspect);

    mog2Params.nMixtures = kMog2Mixtures;
    mog2Params.Tb = static_cast<float>(config.varThreshold);
    mog2Params.TB = clampFloat(static_cast<float>(config.bgRatio), 0.0f, 1.0f);
    mog2Params.Tg = kMog2VarThresholdGen;
    mog2Params.varInit = kMog2VarInit;
    mog2Params.varMin = kMog2VarMin;
    mog2Params.varMax = kMog2VarMax;
    mog2Params.tau = kMog2ShadowTau;
    mog2Params.detectShadows = config.detectShadows ? 1u : 0u;
    morphParams.radius = config.closeKernelSize / 2;

    modelFrameCount = 0;
    clearMog2State();
    frameCounter = 0;
    last.clear();
  }

  std::vector<Pixel> process(const ImageFrame &frame) {
    last.clear();

    CVPixelBufferRef pb = frame.pixelBuffer();
    if (!pb || !device || !queue || !textureCache || !mog2PSO || !dilatePSO || !erodePSO)
      return {};
    if (CVPixelBufferGetPixelFormatType(pb) != kCVPixelFormatType_32BGRA)
      return {};

    frameWidth = static_cast<int>(CVPixelBufferGetWidth(pb));
    frameHeight = static_cast<int>(CVPixelBufferGetHeight(pb));
    if (frameWidth <= 0 || frameHeight <= 0)
      return {};

    scaledWidth = std::max(1, static_cast<int>(std::lround(frameWidth * config.imscale)));
    scaledHeight = std::max(1, static_cast<int>(std::lround(frameHeight * config.imscale)));

    CVMetalTextureRef cvInputTex = nullptr;
    const CVReturn cvResult =
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pb, nullptr,
                                                  MTLPixelFormatBGRA8Unorm, frameWidth, frameHeight, 0, &cvInputTex);
    if (cvResult != kCVReturnSuccess || !cvInputTex)
      return {};

    id<MTLTexture> inputTexture = CVMetalTextureGetTexture(cvInputTex);
    if (!inputTexture) {
      CFRelease(cvInputTex);
      return {};
    }

    if (!ensureWorkingState()) {
      CFRelease(cvInputTex);
      return {};
    }

    ++modelFrameCount;
    const uint64_t lrDenom =
        std::min<uint64_t>(2ULL * std::max<uint64_t>(1ULL, modelFrameCount), static_cast<uint64_t>(config.bgHistory));
    MOG2Params frameParams = mog2Params;
    frameParams.frameWidth = frameWidth;
    frameParams.frameHeight = frameHeight;
    frameParams.scaledWidth = scaledWidth;
    frameParams.scaledHeight = scaledHeight;
    frameParams.alphaT = 1.0f / static_cast<float>(std::max<uint64_t>(1ULL, lrDenom));
    frameParams.alpha1 = 1.0f - frameParams.alphaT;
    frameParams.prune = -frameParams.alphaT * kMog2ComplexityReduction;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
      CFRelease(cvInputTex);
      return {};
    }

    const MTLSize tgSize = makeThreadgroupSize();
    const MTLSize tgCount = makeThreadgroupCount(scaledWidth, scaledHeight, tgSize);

    {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:mog2PSO];
      [enc setTexture:inputTexture atIndex:0];
      [enc setTexture:maskTexture atIndex:1];
      [enc setBuffer:modeMeansBuffer offset:0 atIndex:0];
      [enc setBuffer:modeWeightsBuffer offset:0 atIndex:1];
      [enc setBuffer:modeVariancesBuffer offset:0 atIndex:2];
      [enc setBuffer:modesUsedBuffer offset:0 atIndex:3];
      [enc setBytes:&frameParams length:sizeof(frameParams) atIndex:4];
      [enc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
      [enc endEncoding];
    }

    if (config.closeIterations > 0 && morphParams.radius > 0) {
      for (int i = 0; i < config.closeIterations; ++i) {
        id<MTLComputeCommandEncoder> dilateEnc = [cmd computeCommandEncoder];
        [dilateEnc setComputePipelineState:dilatePSO];
        [dilateEnc setTexture:maskTexture atIndex:0];
        [dilateEnc setTexture:tempMaskTexture atIndex:1];
        [dilateEnc setBytes:&morphParams length:sizeof(morphParams) atIndex:0];
        [dilateEnc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
        [dilateEnc endEncoding];

        id<MTLComputeCommandEncoder> erodeEnc = [cmd computeCommandEncoder];
        [erodeEnc setComputePipelineState:erodePSO];
        [erodeEnc setTexture:tempMaskTexture atIndex:0];
        [erodeEnc setTexture:maskTexture atIndex:1];
        [erodeEnc setBytes:&morphParams length:sizeof(morphParams) atIndex:0];
        [erodeEnc dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];
        [erodeEnc endEncoding];
      }
    }

    [cmd commit];
    [cmd waitUntilCompleted];
    CFRelease(cvInputTex);

    if (cmd.status != MTLCommandBufferStatusCompleted)
      return {};

    const size_t required = static_cast<size_t>(scaledWidth) * static_cast<size_t>(scaledHeight);
    cpuMask.resize(required);
    const MTLRegion region =
        MTLRegionMake2D(0, 0, static_cast<NSUInteger>(scaledWidth), static_cast<NSUInteger>(scaledHeight));
    [maskTexture getBytes:cpuMask.data()
              bytesPerRow:static_cast<NSUInteger>(scaledWidth)
               fromRegion:region
              mipmapLevel:0];

    extractProjectiles(frame.frame);

    std::vector<Pixel> out;
    out.reserve(last.size());
    for (ProjectileFrame &p : last)
      out.push_back(p.center);
    return out;
  }

  const std::vector<ProjectileFrame> &lastProjectiles() const { return last; }

private:
  id<MTLComputePipelineState> createPSO(NSString *name) {
    if (!library)
      return nil;
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn)
      return nil;
    NSError *error = nil;
    return [device newComputePipelineStateWithFunction:fn error:&error];
  }

  bool ensureWorkingState() {
    if (scaledWidth == texWidth && scaledHeight == texHeight && maskTexture && tempMaskTexture && modeMeansBuffer &&
        modeWeightsBuffer && modeVariancesBuffer && modesUsedBuffer) {
      return true;
    }

    texWidth = scaledWidth;
    texHeight = scaledHeight;

    MTLTextureDescriptor *maskDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                           width:static_cast<NSUInteger>(texWidth)
                                                          height:static_cast<NSUInteger>(texHeight)
                                                       mipmapped:NO];
    maskDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    maskDesc.storageMode = MTLStorageModeShared;

    maskTexture = [device newTextureWithDescriptor:maskDesc];
    tempMaskTexture = [device newTextureWithDescriptor:maskDesc];
    if (!maskTexture || !tempMaskTexture)
      return false;

    const size_t pixelCount = static_cast<size_t>(texWidth) * static_cast<size_t>(texHeight);
    const size_t modeCount = pixelCount * static_cast<size_t>(kMog2Mixtures);
    modeMeansBuffer = [device newBufferWithLength:(modeCount * sizeof(float) * 4) options:MTLResourceStorageModeShared];
    modeWeightsBuffer = [device newBufferWithLength:(modeCount * sizeof(float)) options:MTLResourceStorageModeShared];
    modeVariancesBuffer = [device newBufferWithLength:(modeCount * sizeof(float)) options:MTLResourceStorageModeShared];
    modesUsedBuffer = [device newBufferWithLength:(pixelCount * sizeof(uint8_t)) options:MTLResourceStorageModeShared];

    if (!modeMeansBuffer || !modeWeightsBuffer || !modeVariancesBuffer || !modesUsedBuffer)
      return false;

    modelFrameCount = 0;
    clearMog2State();
    return true;
  }

  void clearMog2State() {
    if (!modeMeansBuffer || !modeWeightsBuffer || !modeVariancesBuffer || !modesUsedBuffer || texWidth <= 0 ||
        texHeight <= 0) {
      return;
    }

    const size_t pixelCount = static_cast<size_t>(texWidth) * static_cast<size_t>(texHeight);
    const size_t modeCount = pixelCount * static_cast<size_t>(kMog2Mixtures);
    std::memset([modeMeansBuffer contents], 0, modeCount * sizeof(float) * 4);
    std::memset([modeWeightsBuffer contents], 0, modeCount * sizeof(float));
    std::memset([modeVariancesBuffer contents], 0, modeCount * sizeof(float));
    std::memset([modesUsedBuffer contents], 0, pixelCount * sizeof(uint8_t));
  }

  void extractProjectiles(uint64_t providedFrameCounter) {
    last.clear();
    if (cpuMask.empty() || scaledWidth <= 0 || scaledHeight <= 0)
      return;

    const size_t total = static_cast<size_t>(scaledWidth) * static_cast<size_t>(scaledHeight);
    visited.assign(total, 0);
    queueScratch.clear();
    queueScratch.reserve(2048);

    const bool use8 = (config.connectivity == 8);
    constexpr int k4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    constexpr int k8[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};

    std::vector<ProjectileFrame> temp;
    temp.reserve(64);

    const double invScale = (config.imscale == 0.0 ? 1.0 : 1.0 / config.imscale);

    for (int y = 0; y < scaledHeight; ++y) {
      for (int x = 0; x < scaledWidth; ++x) {
        const int start = y * scaledWidth + x;
        if (visited[start] || cpuMask[start] == 0)
          continue;

        visited[start] = 1;
        queueScratch.clear();
        queueScratch.push_back(start);
        size_t head = 0;

        int area = 0;
        int minX = x, maxX = x, minY = y, maxY = y;

        while (head < queueScratch.size()) {
          const int idx = queueScratch[head++];
          const int cx = idx % scaledWidth;
          const int cy = idx / scaledWidth;
          ++area;
          minX = std::min(minX, cx);
          maxX = std::max(maxX, cx);
          minY = std::min(minY, cy);
          maxY = std::max(maxY, cy);

          if (use8) {
            for (const auto &n : k8) {
              const int nx = cx + n[0];
              const int ny = cy + n[1];
              if (nx < 0 || ny < 0 || nx >= scaledWidth || ny >= scaledHeight)
                continue;
              const int nidx = ny * scaledWidth + nx;
              if (visited[nidx] || cpuMask[nidx] == 0)
                continue;
              visited[nidx] = 1;
              queueScratch.push_back(nidx);
            }
          } else {
            for (const auto &n : k4) {
              const int nx = cx + n[0];
              const int ny = cy + n[1];
              if (nx < 0 || ny < 0 || nx >= scaledWidth || ny >= scaledHeight)
                continue;
              const int nidx = ny * scaledWidth + nx;
              if (visited[nidx] || cpuMask[nidx] == 0)
                continue;
              visited[nidx] = 1;
              queueScratch.push_back(nidx);
            }
          }
        }

        const int w = maxX - minX + 1;
        const int h = maxY - minY + 1;
        if (w <= 0 || h <= 0)
          continue;
        if (area < config.minArea)
          continue;
        const float aspect = static_cast<float>(w) / static_cast<float>(h);
        if (aspect < config.minAspect || aspect > config.maxAspect)
          continue;

        ProjectileFrame p;
        p.bbox.topLeft.x = static_cast<int>(std::lround(static_cast<double>(minX) * invScale));
        p.bbox.topLeft.y = static_cast<int>(std::lround(static_cast<double>(minY) * invScale));
        p.bbox.dimensions.x = std::max(1, static_cast<int>(std::lround(static_cast<double>(w) * invScale)));
        p.bbox.dimensions.y = std::max(1, static_cast<int>(std::lround(static_cast<double>(h) * invScale)));
        p.bbox.area = area;
        p.center.x = p.bbox.topLeft.x + p.bbox.dimensions.x / 2;
        p.center.y = p.bbox.topLeft.y + p.bbox.dimensions.y / 2;
        p.frame = providedFrameCounter;
        temp.push_back(p);
      }
    }

    std::sort(temp.begin(), temp.end(),
              [](const ProjectileFrame &a, const ProjectileFrame &b) { return a.bbox.area > b.bbox.area; });
    if (static_cast<int>(temp.size()) > config.maxProjectiles)
      temp.resize(config.maxProjectiles);

    last = std::move(temp);
    ++frameCounter;
  }

  DetectorConfig config;
  MOG2Params mog2Params;
  MorphParams morphParams;

  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> mog2PSO = nil;
  id<MTLComputePipelineState> dilatePSO = nil;
  id<MTLComputePipelineState> erodePSO = nil;

  CVMetalTextureCacheRef textureCache = nullptr;

  id<MTLTexture> maskTexture = nil;
  id<MTLTexture> tempMaskTexture = nil;
  id<MTLBuffer> modeMeansBuffer = nil;
  id<MTLBuffer> modeWeightsBuffer = nil;
  id<MTLBuffer> modeVariancesBuffer = nil;
  id<MTLBuffer> modesUsedBuffer = nil;

  int texWidth = 0;
  int texHeight = 0;
  int frameWidth = 0;
  int frameHeight = 0;
  int scaledWidth = 0;
  int scaledHeight = 0;
  uint64_t modelFrameCount = 0;
  uint64_t frameCounter = 0;

  std::vector<uint8_t> cpuMask;
  std::vector<uint8_t> visited;
  std::vector<int> queueScratch;
  std::vector<ProjectileFrame> last;
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

const std::vector<ProjectileFrame> &Detector::lastProjectiles() const {
  static const std::vector<ProjectileFrame> kEmpty;
  if (!_impl)
    return kEmpty;
  return _impl->lastProjectiles();
}

void Detector::applyConfig(const DetectorConfig &config) {
  if (_impl)
    _impl->applyConfig(config);
}

void drawPoint(ImageFrame &frame, const Pixel &point, int radius) {
  LockedFrameView locked(frame.pixelBuffer(), 0);
  if (!locked.valid())
    return;
  drawPointIntoBGRA(locked.base(), locked.width(), locked.height(), locked.stride(), point, radius);
}

void drawPoints(ImageFrame &frame, const std::vector<Pixel> &points, int radius) {
  if (points.empty())
    return;
  LockedFrameView locked(frame.pixelBuffer(), 0);
  if (!locked.valid())
    return;
  for (const Pixel &p : points)
    drawPointIntoBGRA(locked.base(), locked.width(), locked.height(), locked.stride(), p, radius);
}

void drawProjectiles(ImageFrame &frame, const std::vector<ProjectileFrame> &projectiles, int boxThickness,
                     int centerRadius) {
  if (projectiles.empty())
    return;
  LockedFrameView locked(frame.pixelBuffer(), 0);
  if (!locked.valid())
    return;

  for (const ProjectileFrame &p : projectiles) {
    const RectI bbox{p.bbox.topLeft.x, p.bbox.topLeft.y, p.bbox.dimensions.x, p.bbox.dimensions.y};
    drawRectIntoBGRA(locked.base(), locked.width(), locked.height(), locked.stride(), bbox, boxThickness);
    drawPointIntoBGRA(locked.base(), locked.width(), locked.height(), locked.stride(), p.center, centerRadius);
  }
}

} // namespace pd
