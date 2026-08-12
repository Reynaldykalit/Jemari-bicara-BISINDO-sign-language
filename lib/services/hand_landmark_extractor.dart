import 'package:camera/camera.dart';
import 'package:flutter/services.dart';

class HandLandmarkExtractor {
  static const MethodChannel _channel = MethodChannel(
    'jemari_bicara/hand_landmarker',
  );

  Future<Map<String, dynamic>?> processFrame(
    CameraImage image,
    int rotation, {
    double confThresh = 0.17,
    double startMotionThresh = 0.005,
    double endMotionThresh = 0.002,
    int totalCameraFrames = 0,
    bool isFrontCamera = false,
    bool enableYoloSkip = true,
    int yoloSkipLandmarkThreshold = 2,
    int yoloRerunMissThreshold = 2,
    int yoloRerunIntervalFrames = 15,
    double roiPaddingScale = 1.3,
  }) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'processFrame',
        {
          'width': image.width,
          'height': image.height,
          'format': image.format.group.name,
          'rotation': rotation,
          'confThresh': confThresh,
          'startMotionThresh': startMotionThresh,
          'endMotionThresh': endMotionThresh,
          'minGestureLength': 10,
          'maxGestureLength': 90,
          'cameraTime': DateTime.now().millisecondsSinceEpoch,
          'totalCameraFrames': totalCameraFrames,
          'isFrontCamera': isFrontCamera,
          'enableYoloSkip': enableYoloSkip,
          'yoloSkipLandmarkThreshold': yoloSkipLandmarkThreshold,
          'yoloRerunMissThreshold': yoloRerunMissThreshold,
          'yoloRerunIntervalFrames': yoloRerunIntervalFrames,
          'roiPaddingScale': roiPaddingScale,
          'planes': image.planes
              .map(
                (plane) => {
                  'bytes': plane.bytes,
                  'bytesPerRow': plane.bytesPerRow,
                  'bytesPerPixel': plane.bytesPerPixel,
                  'height': plane.height,
                  'width': plane.width,
                },
              )
              .toList(growable: false),
        },
      );

      return result;
    } on MissingPluginException {
      return null;
    } on PlatformException {
      return null;
    }
  }

  Future<void> clearBuffer() async {
    try {
      await _channel.invokeMethod<void>('clearBuffer');
    } catch (_) {}
  }

  Future<Map<String, dynamic>?> getNativeStatus() async {
    try {
      return await _channel.invokeMapMethod<String, dynamic>('getNativeStatus');
    } catch (_) {
      return null;
    }
  }
}
