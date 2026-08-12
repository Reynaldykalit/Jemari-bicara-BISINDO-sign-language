import 'dart:typed_data';

class PipelineMetrics {
  final String status;
  final int timestamp;
  
  // Camera
  final int cameraWidth;
  final int cameraHeight;
  final int cameraRotation;
  final String cameraFacing;

  // Image Conversion
  final double conversionTimeMs;
  final int yuvWidth;
  final int yuvHeight;
  final String pixelFormat;

  // Rotation
  final double rotationTimeMs;
  final int originalWidth;
  final int originalHeight;
  final int rotatedWidth;
  final int rotatedHeight;

  // YOLO
  final double yoloTimeMs;
  final double yoloConfidence;
  final int yoloClassId;
  final List<double> yoloBox;

  // Crop ROI
  final double cropTimeMs;
  final int cropWidth;
  final int cropHeight;

  // Resize ROI
  final int resizedWidth;
  final int resizedHeight;
  final String resizedInterpolation;

  // ROI Bytes
  final Uint8List? roiBytes;

  // MediaPipe
  final double mpTimeMs;
  final int mpLandmarkCount;
  final List<double> rawLandmarks;
  final double landmarkMin;
  final double landmarkMax;
  final double landmarkAvg;
  final bool hasNaN;
  final bool hasInfinity;

  // Landmark Buffer
  final int sequenceLength;
  final String landmarkBufferState;
  final double windowMotion;

  // Tensor Generation
  final List<int> tensorShape;
  final String tensorDataType;
  final List<double> tensorFirstFrame;
  final List<double> tensorLastFrame;

  // LSTM
  final double lstmTimeMs;
  final int lstmOutputSize;
  final List<double> probabilities;

  // Predictions
  final int labelIndex;
  final double confidence;
  final double totalTimeMs;

  // Phase 1 Profiling Additions
  final double threadCreationTimeMs;
  final bool yoloSkipped;
  final bool roiSmoothed;

  // Phase 2 Profiling Additions
  final double acquisitionDelayMs;
  final double yoloCoverage;
  final double motionToYoloDelayMs;
  final double motionToRoiDelayMs;
  final double motionToMpDelayMs;
  final double motionToBufferDelayMs;

  PipelineMetrics({
    required this.status,
    required this.timestamp,
    required this.cameraWidth,
    required this.cameraHeight,
    required this.cameraRotation,
    required this.cameraFacing,
    required this.conversionTimeMs,
    required this.yuvWidth,
    required this.yuvHeight,
    required this.pixelFormat,
    required this.rotationTimeMs,
    required this.originalWidth,
    required this.originalHeight,
    required this.rotatedWidth,
    required this.rotatedHeight,
    required this.yoloTimeMs,
    required this.yoloConfidence,
    required this.yoloClassId,
    required this.yoloBox,
    required this.cropTimeMs,
    required this.cropWidth,
    required this.cropHeight,
    required this.resizedWidth,
    required this.resizedHeight,
    required this.resizedInterpolation,
    this.roiBytes,
    required this.mpTimeMs,
    required this.mpLandmarkCount,
    required this.rawLandmarks,
    required this.landmarkMin,
    required this.landmarkMax,
    required this.landmarkAvg,
    required this.hasNaN,
    required this.hasInfinity,
    required this.sequenceLength,
    required this.landmarkBufferState,
    required this.windowMotion,
    required this.tensorShape,
    required this.tensorDataType,
    required this.tensorFirstFrame,
    required this.tensorLastFrame,
    required this.lstmTimeMs,
    required this.lstmOutputSize,
    required this.probabilities,
    required this.labelIndex,
    required this.confidence,
    required this.totalTimeMs,
    required this.threadCreationTimeMs,
    required this.yoloSkipped,
    required this.roiSmoothed,
    required this.acquisitionDelayMs,
    required this.yoloCoverage,
    required this.motionToYoloDelayMs,
    required this.motionToRoiDelayMs,
    required this.motionToMpDelayMs,
    required this.motionToBufferDelayMs,
  });

  factory PipelineMetrics.fromMap(Map<String, dynamic> map) {
    return PipelineMetrics(
      status: map['status'] as String? ?? 'collecting',
      timestamp: map['timestamp'] as int? ?? 0,
      cameraWidth: map['cameraWidth'] as int? ?? 0,
      cameraHeight: map['cameraHeight'] as int? ?? 0,
      cameraRotation: map['cameraRotation'] as int? ?? 0,
      cameraFacing: map['cameraFacing'] as String? ?? 'front',
      conversionTimeMs: (map['conversionTimeMs'] as num?)?.toDouble() ?? 0.0,
      yuvWidth: map['yuvWidth'] as int? ?? 0,
      yuvHeight: map['yuvHeight'] as int? ?? 0,
      pixelFormat: map['pixelFormat'] as String? ?? '',
      rotationTimeMs: (map['rotationTimeMs'] as num?)?.toDouble() ?? 0.0,
      originalWidth: map['originalWidth'] as int? ?? 0,
      originalHeight: map['originalHeight'] as int? ?? 0,
      rotatedWidth: map['rotatedWidth'] as int? ?? 0,
      rotatedHeight: map['rotatedHeight'] as int? ?? 0,
      yoloTimeMs: (map['yoloTimeMs'] as num?)?.toDouble() ?? 0.0,
      yoloConfidence: (map['yoloConfidence'] as num?)?.toDouble() ?? 0.0,
      yoloClassId: map['yoloClassId'] as int? ?? 0,
      yoloBox: (map['yoloBox'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList() ?? [],
      cropTimeMs: (map['cropTimeMs'] as num?)?.toDouble() ?? 0.0,
      cropWidth: map['cropWidth'] as int? ?? 0,
      cropHeight: map['cropHeight'] as int? ?? 0,
      resizedWidth: map['resizedWidth'] as int? ?? 0,
      resizedHeight: map['resizedHeight'] as int? ?? 0,
      resizedInterpolation: map['resizedInterpolation'] as String? ?? '',
      roiBytes: map['roiBytes'] as Uint8List?,
      mpTimeMs: (map['mpTimeMs'] as num?)?.toDouble() ?? 0.0,
      mpLandmarkCount: map['mpLandmarkCount'] as int? ?? 0,
      rawLandmarks: (map['rawLandmarks'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList() ?? [],
      landmarkMin: (map['landmarkMin'] as num?)?.toDouble() ?? 0.0,
      landmarkMax: (map['landmarkMax'] as num?)?.toDouble() ?? 0.0,
      landmarkAvg: (map['landmarkAvg'] as num?)?.toDouble() ?? 0.0,
      hasNaN: map['hasNaN'] as bool? ?? false,
      hasInfinity: map['hasInfinity'] as bool? ?? false,
      sequenceLength: map['sequenceLength'] as int? ?? 0,
      landmarkBufferState: map['landmarkBufferState'] as String? ?? '',
      windowMotion: (map['windowMotion'] as num?)?.toDouble() ?? 0.0,
      tensorShape: (map['tensorShape'] as List<dynamic>?)?.map((e) => e as int).toList() ?? [],
      tensorDataType: map['tensorDataType'] as String? ?? '',
      tensorFirstFrame: (map['tensorFirstFrame'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList() ?? [],
      tensorLastFrame: (map['tensorLastFrame'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList() ?? [],
      lstmTimeMs: (map['lstmTimeMs'] as num?)?.toDouble() ?? 0.0,
      lstmOutputSize: map['lstmOutputSize'] as int? ?? 0,
      probabilities: (map['probabilities'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList() ?? [],
      labelIndex: map['labelIndex'] as int? ?? -1,
      confidence: (map['confidence'] as num?)?.toDouble() ?? 0.0,
      totalTimeMs: (map['totalTimeMs'] as num?)?.toDouble() ?? 0.0,
      threadCreationTimeMs: (map['threadCreationTimeMs'] as num?)?.toDouble() ?? 0.0,
      yoloSkipped: map['yoloSkipped'] as bool? ?? false,
      roiSmoothed: map['roiSmoothed'] as bool? ?? false,
      acquisitionDelayMs: (map['acquisitionDelayMs'] as num?)?.toDouble() ?? -1.0,
      yoloCoverage: (map['yoloCoverage'] as num?)?.toDouble() ?? 0.0,
      motionToYoloDelayMs: (map['motionToYoloDelayMs'] as num?)?.toDouble() ?? -1.0,
      motionToRoiDelayMs: (map['motionToRoiDelayMs'] as num?)?.toDouble() ?? -1.0,
      motionToMpDelayMs: (map['motionToMpDelayMs'] as num?)?.toDouble() ?? -1.0,
      motionToBufferDelayMs: (map['motionToBufferDelayMs'] as num?)?.toDouble() ?? -1.0,
    );
  }
}
