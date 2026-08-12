import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';

import '../models/pipeline_metrics.dart';
import 'hand_landmark_extractor.dart';
import 'landmark_sequence_buffer.dart';

class BisindoModelService {
  BisindoModelService({
    HandLandmarkExtractor? landmarkExtractor,
    LandmarkSequenceBuffer? sequenceBuffer,
  }) : _landmarkExtractor = landmarkExtractor ?? HandLandmarkExtractor();

  static const String labelMapAsset = 'assets/models/label_map.json';

  final HandLandmarkExtractor _landmarkExtractor;
  Map<int, String> _labels = {};
  Map<int, String> get labels => _labels;

  String statusMessage = 'Model belum dimuat';
  bool get isReady => _labels.isNotEmpty;

  Future<void> load() async {
    try {
      final labelJson = await rootBundle.loadString(labelMapAsset);
      _labels = _parseLabels(labelJson);

      print('Checking native models status...');
      final nativeStatus = await _landmarkExtractor.getNativeStatus();
      if (nativeStatus != null) {
        final yoloReady = nativeStatus['yoloReady'] as bool? ?? false;
        final landmarkerReady =
            nativeStatus['landmarkerReady'] as bool? ?? false;
        final lstmReady = nativeStatus['lstmReady'] as bool? ?? false;

        if (!yoloReady || !landmarkerReady || !lstmReady) {
          final yoloErr = nativeStatus['yoloError'] as String? ?? 'YOLO failed';
          final lmErr =
              nativeStatus['landmarkerError'] as String? ?? 'MediaPipe failed';
          final lstmErr = nativeStatus['lstmError'] as String? ?? 'LSTM failed';
          throw Exception(
            'Native models load failed: YOLO=[$yoloReady, err:$yoloErr], MP=[$landmarkerReady, err:$lmErr], LSTM=[$lstmReady, err:$lstmErr]',
          );
        }
      }

      statusMessage = 'Model siap';
      print('Model configuration and labels loaded successfully');
    } catch (error, stackTrace) {
      statusMessage = 'Gagal memuat model: $error';
      print('Error loading Bisindo model: $error');
      print(stackTrace);
      rethrow;
    }
  }

  Future<PipelineMetrics?> processCameraImage(
    CameraImage image,
    int rotation, {
    double confThresh = 0.17,
    double startMotionThresh = 0.012,
    double endMotionThresh = 0.006,
    int totalCameraFrames = 0,
    bool isFrontCamera = false,
    bool enableYoloSkip = true,
    int yoloSkipLandmarkThreshold = 2,
    int yoloRerunMissThreshold = 2,
    int yoloRerunIntervalFrames = 15,
    double roiPaddingScale = 1.3,
  }) async {
    if (!isReady) {
      return null;
    }

    final result = await _landmarkExtractor.processFrame(
      image,
      rotation,
      confThresh: confThresh,
      startMotionThresh: startMotionThresh,
      endMotionThresh: endMotionThresh,
      totalCameraFrames: totalCameraFrames,
      isFrontCamera: isFrontCamera,
      enableYoloSkip: enableYoloSkip,
      yoloSkipLandmarkThreshold: yoloSkipLandmarkThreshold,
      yoloRerunMissThreshold: yoloRerunMissThreshold,
      yoloRerunIntervalFrames: yoloRerunIntervalFrames,
      roiPaddingScale: roiPaddingScale,
    );
    if (result == null) {
      statusMessage = 'Tangan belum terdeteksi';
      return null;
    }

    final metrics = PipelineMetrics.fromMap(result);

    if (metrics.status == 'error') {
      statusMessage =
          'Error native: ${result['message'] ?? 'Unknown native error'}';
    } else if (metrics.status == 'READY') {
      statusMessage = 'Model Siap';
    } else if (metrics.status == 'GESTURE_COLLECTION') {
      statusMessage = 'Mendeteksi...';
    } else if (metrics.status == 'BUFFER_FROZEN' || metrics.status == 'LSTM_INFERENCE') {
      statusMessage = 'Memproses...';
    } else if (metrics.status == 'DISPLAY_RESULT' || metrics.status == 'WAIT_NEXT_GESTURE') {
      statusMessage = 'Hasil Ditemukan';
    }

    return metrics;
  }

  void clear() {
    _landmarkExtractor.clearBuffer();
  }

  void close() {
    // No Dart interpreter to close
  }

  Map<int, String> _parseLabels(String jsonText) {
    final data = jsonDecode(jsonText) as Map<String, dynamic>;
    final idx2label = data['idx2label'] as Map<String, dynamic>?;
    final label2word = data['label2word'] as Map<String, dynamic>?;

    if (idx2label == null) {
      return {};
    }

    return idx2label.map((key, value) {
      final labelStr = value.toString();
      final word = label2word != null ? label2word[labelStr] : null;
      return MapEntry(int.parse(key), (word ?? labelStr).toString());
    });
  }
}
