import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../data/dummy_data.dart';
import '../models/history_model.dart';
import '../models/pipeline_metrics.dart';
import '../services/bisindo_model_service.dart';
import '../services/ood_filter_service.dart';
import '../theme/app_theme.dart';

class RealtimeScreen extends StatefulWidget {
  const RealtimeScreen({super.key});

  @override
  State<RealtimeScreen> createState() => _RealtimeScreenState();
}

class _RealtimeScreenState extends State<RealtimeScreen> {
  CameraController? cameraController;
  List<CameraDescription> cameras = [];
  CameraLensDirection _lensDirection = CameraLensDirection.back;

  final BisindoModelService modelService = BisindoModelService();

  String currentTranslation = '';
  String modelStatus = 'Memuat model...';
  double confidence = 0.0;

  bool isCameraReady = false;
  bool isCameraOn = true;
  bool isPaused = false;
  bool isProcessingFrame = false;
  bool isSwitchingCamera = false;
  bool hasCameraPermission = false;
  bool hasDisplayedPrediction = false;

  // Latency & frame acquisition diagnostic variables
  int totalCameraFrames = 0;
  int droppedBeforeYolo = 0;
  DateTime? lastFpsLogTime;
  int fpsFrameCount = 0;
  List<double>? detectedHandBox;

  PipelineMetrics? latestMetrics;
  bool isDeveloperMode = false;
  double confThresh = 0.17;
  double startMotionThresh = 0.007;
  double endMotionThresh = 0.004;

  // Phase 3 Parameters
  bool enableYoloSkip = true;
  double yoloSkipLandmarkThreshold = 2.0;
  double yoloRerunMissThreshold = 2.0;
  double yoloRerunIntervalFrames = 15.0;

  // Phase 5 Parameters
  double roiPaddingScale = 1.45;

  DateTime? lastFrameTime;
  DateTime? lastPredictionTime;
  double liveFPS = 0.0;

  Timer? _clearResultTimer;
  static const int autoClearDurationSeconds = 5;

  static const Color mainBlue = Color(0xFF76AEE6);

  @override
  void initState() {
    super.initState();
    loadModel();
    initCamera();
  }

  void _startAutoClearTimer() {
    _clearResultTimer?.cancel();
    _clearResultTimer = Timer(const Duration(seconds: autoClearDurationSeconds), () {
      if (mounted) {
        setState(() {
          currentTranslation = '';
          confidence = 0.0;
          modelStatus = 'Model Siap';
          hasDisplayedPrediction = false;
        });
      }
    });
  }

  Future<void> loadModel() async {
    try {
      await modelService.load();
      if (!mounted) return;
      setState(() {
        modelStatus = modelService.statusMessage;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        modelStatus = modelService.statusMessage;
      });
    }
  }

  Future<void> initCamera() async {
    final status = await Permission.camera.request();

    if (!status.isGranted) {
      if (!mounted) return;

      setState(() {
        hasCameraPermission = false;
        isCameraReady = false;
        isCameraOn = false;
        isPaused = true;
      });
      return;
    }

    hasCameraPermission = true;

    cameras = await availableCameras();

    if (cameras.isEmpty) {
      if (!mounted) return;

      setState(() {
        isCameraReady = false;
        isCameraOn = false;
        isPaused = true;
      });
      return;
    }

    final selectedCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == _lensDirection,
      orElse: () => cameras.first,
    );

    final controller = CameraController(
      selectedCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    cameraController = controller;

    try {
      await controller.initialize();
    } catch (_) {
      if (!mounted) return;

      setState(() {
        cameraController = null;
        isCameraReady = false;
        isCameraOn = false;
        isPaused = true;
      });
      return;
    }

    if (!mounted || cameraController != controller) {
      await controller.dispose();
      return;
    }

    setState(() {
      isCameraReady = true;
      isCameraOn = true;
      isPaused = false;
    });

    startCameraStream();
  }

  Future<void> turnOffCamera() async {
    if (isSwitchingCamera) return;

    setState(() {
      isSwitchingCamera = true;
      isCameraOn = false;
      isPaused = true;
    });

    final controller = cameraController;
    cameraController = null;

    try {
      if (controller != null && controller.value.isStreamingImages) {
        await controller.stopImageStream().timeout(
          const Duration(seconds: 2),
          onTimeout: () {},
        );
      }
    } catch (_) {}

    while (isProcessingFrame) {
      await Future.delayed(const Duration(milliseconds: 50));
    }

    try {
      await controller?.dispose().timeout(
        const Duration(seconds: 2),
        onTimeout: () {},
      );
    } catch (_) {}

    if (!mounted) return;

    setState(() {
      isCameraReady = false;
      currentTranslation = '';
      confidence = 0;
      isSwitchingCamera = false;
    });
  }

  Future<void> turnOnCamera() async {
    if (isSwitchingCamera) return;

    setState(() {
      isSwitchingCamera = true;
      isCameraOn = true;
      isPaused = false;
      isCameraReady = false;
    });

    await initCamera();

    if (!mounted) return;

    setState(() {
      isSwitchingCamera = false;
    });
  }

  Future<void> toggleCamera() async {
    if (isSwitchingCamera) return;

    if (isCameraOn) {
      await turnOffCamera();
    } else {
      await turnOnCamera();
    }
  }

  Future<void> switchCamera() async {
    if (cameras.length < 2 || isSwitchingCamera) return;

    final newDirection = _lensDirection == CameraLensDirection.front
        ? CameraLensDirection.back
        : CameraLensDirection.front;

    _lensDirection = newDirection;

    if (isCameraOn) {
      setState(() {
        isSwitchingCamera = true;
      });

      final controller = cameraController;
      cameraController = null;

      try {
        if (controller != null && controller.value.isStreamingImages) {
          await controller.stopImageStream().timeout(
            const Duration(seconds: 2),
            onTimeout: () {},
          );
        }
      } catch (_) {}

      while (isProcessingFrame) {
        await Future.delayed(const Duration(milliseconds: 50));
      }

      try {
        await controller?.dispose().timeout(
          const Duration(seconds: 2),
          onTimeout: () {},
        );
      } catch (_) {}

      if (!mounted) return;

      await initCamera();

      if (!mounted) return;
      setState(() {
        isSwitchingCamera = false;
      });
    } else {
      setState(() {});
    }
  }

  void startCameraStream() {
    final controller = cameraController;

    if (controller == null || controller.value.isStreamingImages) return;

    totalCameraFrames = 0;
    droppedBeforeYolo = 0;
    fpsFrameCount = 0;
    lastFpsLogTime = DateTime.now();

    controller.startImageStream((CameraImage image) async {
      totalCameraFrames++;
      fpsFrameCount++;

      final now = DateTime.now();
      if (lastFpsLogTime == null) {
        lastFpsLogTime = now;
      } else if (now.difference(lastFpsLogTime!).inSeconds >= 1) {
        final elapsed = now.difference(lastFpsLogTime!).inMilliseconds / 1000.0;
        final callbackFps = fpsFrameCount / elapsed;
        debugPrint(
          'PIPELINE_VALIDATION: [Stage 1] Camera Callback FPS: ${callbackFps.toStringAsFixed(1)} | Total Frames: $totalCameraFrames | Dropped: $droppedBeforeYolo',
        );
        fpsFrameCount = 0;
        lastFpsLogTime = now;
      }

      if (!isCameraOn || isPaused || isProcessingFrame || isSwitchingCamera) {
        droppedBeforeYolo++;
        return;
      }

      isProcessingFrame = true;

      try {
        await processFrameWithModel(image);
      } finally {
        isProcessingFrame = false;
      }
    });
  }

  Future<void> processFrameWithModel(CameraImage image) async {
    if (!mounted || !isCameraOn || isPaused || isSwitchingCamera) return;
    if (!modelService.isReady) return;

    // 1. Calculate live FPS
    final now = DateTime.now();
    if (lastFrameTime != null) {
      final diff = now.difference(lastFrameTime!).inMilliseconds;
      if (diff > 0) {
        liveFPS = 1000.0 / diff;
      }
    }
    lastFrameTime = now;

    final rotation = cameraController?.description.sensorOrientation ?? 0;
    final result = await modelService.processCameraImage(
      image,
      rotation,
      confThresh: confThresh,
      startMotionThresh: startMotionThresh,
      endMotionThresh: endMotionThresh,
      totalCameraFrames: totalCameraFrames,
      isFrontCamera: _lensDirection == CameraLensDirection.front,
      enableYoloSkip: enableYoloSkip,
      yoloSkipLandmarkThreshold: yoloSkipLandmarkThreshold.toInt(),
      yoloRerunMissThreshold: yoloRerunMissThreshold.toInt(),
      yoloRerunIntervalFrames: yoloRerunIntervalFrames.toInt(),
      roiPaddingScale: roiPaddingScale,
    );

    if (!mounted || !isCameraOn || isPaused || isSwitchingCamera) return;

    setState(() {
      latestMetrics = result;
      modelStatus = modelService.statusMessage;

      if (result != null) {
        // Draw YOLO box if hand is detected by MediaPipe / YOLO
        if (result.mpLandmarkCount > 0 &&
            result.yoloConfidence >= confThresh &&
            result.status != 'DISPLAY_RESULT' &&
            result.status != 'WAIT_NEXT_GESTURE') {
          detectedHandBox = result.yoloBox;
        } else {
          detectedHandBox = null;
        }

        if (result.status == 'WAIT_NEXT_GESTURE' || result.status == 'DISPLAY_RESULT') {
          if (!hasDisplayedPrediction) {
            final filterResult = OodFilterService.evaluatePrediction(
              labelIndex: result.labelIndex,
              confidence: result.confidence,
              rawLandmarks: result.rawLandmarks,
              probabilities: result.probabilities,
              tensorLastFrame: result.tensorLastFrame,
            );

            if (!filterResult.isAccepted) {
              if (filterResult.statusMessage == 'Bukan gerakan BISINDO') {
                currentTranslation = 'Bukan gerakan BISINDO';
                confidence = 0.0;
                modelStatus = 'Gerakan kurang jelas';
                _startAutoClearTimer();
              } else {
                // PASSBY_MOTION: Jika belum ada teks yang tampil, bersihkan. Jika sudah ada teks, biarkan timer 5 detik berjalan!
                if (currentTranslation.isEmpty) {
                  _clearResultTimer?.cancel();
                  confidence = 0.0;
                  modelStatus = 'Model Siap';
                }
              }
              debugPrint(
                '[PIPELINE] Gesture Rejected (${filterResult.statusMessage}): ${filterResult.rejectReason}',
              );
            } else {
              final finalLabelIndex =
                  filterResult.overrideLabelIndex ?? result.labelIndex;
              final label =
                  modelService.labels[finalLabelIndex] ??
                  'label$finalLabelIndex';
              currentTranslation = label;
              confidence = result.confidence;
              modelStatus = 'Hasil Ditemukan';

              // Otomatis simpan ke riwayat tanpa tombol manual
              if (label.isNotEmpty &&
                  (DummyData.history.isEmpty ||
                      DummyData.history.first.text != label)) {
                DummyData.history.insert(
                  0,
                  HistoryModel(
                    text: label,
                    timestamp: DateTime.now(),
                  ),
                );
              }

              _startAutoClearTimer();
              debugPrint(
                '[PIPELINE] Prediction Accepted ($label) - Confidence: ${result.confidence} (${filterResult.rejectReason})',
              );
            }
            hasDisplayedPrediction = true;
          }
        } else if (result.status == 'GESTURE_COLLECTION') {
          hasDisplayedPrediction = false;
          modelStatus = 'Mengumpulkan Gesture';
        } else if (result.status == 'READY') {
          modelStatus = 'Model Siap';
        }
      } else {
        detectedHandBox = null;
      }
    });
  }

  void saveHistory() {
    if (currentTranslation.isEmpty) return;

    DummyData.history.insert(
      0,
      HistoryModel(text: currentTranslation, timestamp: DateTime.now()),
    );

    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Terjemahan disimpan')));
  }

  void clearText() {
    modelService.clear();
    setState(() {
      currentTranslation = '';
      confidence = 0;
      latestMetrics = null;
      hasDisplayedPrediction = false;
    });
  }

  @override
  void dispose() {
    _clearResultTimer?.cancel();
    final controller = cameraController;
    cameraController = null;

    if (controller != null) {
      if (controller.value.isStreamingImages) {
        controller.stopImageStream().catchError((_) {});
      }
      controller.dispose();
    }

    modelService.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: isDark ? AppTheme.darkBackground : const Color(0xFFE6F2FF),
      body: SafeArea(
        child: Column(
          children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.fromLTRB(18, 14, 22, 18),
              decoration: BoxDecoration(
                color: isDark ? AppTheme.darkCard : mainBlue,
                borderRadius: const BorderRadius.vertical(
                  bottom: Radius.circular(28),
                ),
              ),
              child: Row(
                children: [
                  IconButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    icon: const Icon(Icons.arrow_back, color: Colors.white),
                  ),
                  const Expanded(
                    child: Text(
                      'Halaman Terjemahan',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 19,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                  const SizedBox(width: 48),
                ],
              ),
            ),

            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    Container(
                      height: MediaQuery.of(context).size.height * 0.52,
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: isDark ? AppTheme.darkCard : Colors.white,
                        border: Border.all(
                          color: isDark ? const Color(0xFF475569) : mainBlue,
                          width: 2,
                        ),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(14),
                        child: Stack(
                          children: [
                            Positioned.fill(child: buildCameraPreview()),
                            if (isCameraOn &&
                                isCameraReady &&
                                !isSwitchingCamera &&
                                cameras.length >= 2)
                              Positioned(
                                bottom: 14,
                                right: 14,
                                child: Material(
                                  color: Colors.black.withValues(alpha: 0.5),
                                  borderRadius: BorderRadius.circular(30),
                                  child: IconButton(
                                    icon: const Icon(
                                      Icons.switch_camera_rounded,
                                      color: Colors.white,
                                      size: 26,
                                    ),
                                    onPressed: switchCamera,
                                    tooltip: 'Ganti Kamera',
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                    ),

                    const SizedBox(height: 10),

                    _buildCompactResultCard(),

                    const SizedBox(height: 10),

                    Row(
                      children: [
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: isCameraOn && !isSwitchingCamera
                                ? () {
                                    setState(() {
                                      isPaused = !isPaused;
                                    });
                                  }
                                : null,
                            icon: Icon(
                              isPaused ? Icons.play_arrow : Icons.pause,
                              color: isDark ? Colors.white : Colors.black,
                              size: 18,
                            ),
                            label: Text(isPaused ? 'Lanjutkan' : 'Jeda'),
                            style: OutlinedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              foregroundColor: isDark ? Colors.white : Colors.black,
                              backgroundColor: isDark ? AppTheme.darkCard : Colors.white,
                              disabledForegroundColor: Colors.grey,
                              side: BorderSide(
                                color: isDark ? const Color(0xFF475569) : mainBlue,
                              ),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: clearText,
                            icon: Icon(
                              Icons.delete_outline,
                              color: isDark ? Colors.white : Colors.black,
                              size: 18,
                            ),
                            label: const Text('Hapus'),
                            style: OutlinedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              foregroundColor: isDark ? Colors.white : Colors.black,
                              backgroundColor: isDark ? AppTheme.darkCard : Colors.white,
                              side: BorderSide(
                                color: isDark ? const Color(0xFF475569) : mainBlue,
                              ),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCompactResultCard() {
    final confidencePercent = (confidence * 100).toStringAsFixed(0);
    final isDark = Theme.of(context).brightness == Brightness.dark;

    final isUnrecognized =
        currentTranslation.startsWith('Gesture kurang jelas') ||
        currentTranslation.startsWith('Gesture tidak dikenali') ||
        currentTranslation.startsWith('Bukan gerakan BISINDO');

    final isModelReady = modelStatus.contains('Siap') ||
        modelStatus.contains('READY') ||
        modelStatus == 'Model Siap';
    final isRecording = modelStatus.contains('Mengumpulkan') ||
        modelStatus.contains('Gerakan') ||
        modelStatus.contains('Merekam') ||
        modelStatus.contains('Menangkap') ||
        modelStatus.contains('Mendeteksi');
    final isProcessing = modelStatus.contains('Memproses') ||
        modelStatus.contains('Menerjemahkan') ||
        modelStatus.contains('LSTM');

    String badgeText;
    Color badgeBgColor;
    Color badgeTextColor;

    if (isProcessing) {
      badgeText = '[ ooo Memproses... ]';
      badgeBgColor = isDark ? const Color(0xFF1E3A8A) : const Color(0xFFDBEAFE);
      badgeTextColor = isDark
          ? const Color(0xFF93C5FD)
          : const Color(0xFF2563EB);
    } else if (isRecording) {
      badgeText = '[ ||| Mendeteksi... ]';
      badgeBgColor = isDark ? const Color(0xFF78350F) : const Color(0xFFFEF3C7);
      badgeTextColor = isDark
          ? const Color(0xFFFDE68A)
          : const Color(0xFFD97706);
    } else if (isModelReady) {
      badgeText = '[ ⭕ Model Siap ]';
      badgeBgColor = isDark ? const Color(0xFF334155) : const Color(0xFFF1F5F9);
      badgeTextColor = isDark
          ? const Color(0xFF94A3B8)
          : const Color(0xFF475569);
    } else if (currentTranslation.isNotEmpty && !isUnrecognized) {
      badgeText = '[ ✔️ Selesai ]';
      badgeBgColor = isDark ? const Color(0xFF065F46) : const Color(0xFFDCFCE7);
      badgeTextColor = isDark
          ? const Color(0xFF6EE7B7)
          : const Color(0xFF16A34A);
    } else {
      badgeText = '[ ⭕ Model Siap ]';
      badgeBgColor = isDark ? const Color(0xFF334155) : const Color(0xFFF1F5F9);
      badgeTextColor = isDark
          ? const Color(0xFF94A3B8)
          : const Color(0xFF475569);
    }

    // Determine body display text & confidence placement
    Widget bodyContent;

    if (currentTranslation.isNotEmpty) {
      if (isUnrecognized) {
        // Gesture tidak dikenali -> Tanpa persentase confidence
        bodyContent = Text(
          currentTranslation,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Colors.orange.shade800,
          ),
        );
      } else {
        // Teks Hasil + Persentase Confidence berdampingan (e.g. Halo  90%)
        bodyContent = Row(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.baseline,
          textBaseline: TextBaseline.alphabetic,
          children: [
            Flexible(
              child: Text(
                currentTranslation,
                style: TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.w800,
                  color: isDark ? Colors.white : const Color(0xFF0F172A),
                ),
              ),
            ),
            const SizedBox(width: 10),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: isDark
                    ? const Color(0xFF065F46)
                    : const Color(0xFFDCFCE7),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                '$confidencePercent%',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: isDark
                      ? const Color(0xFF6EE7B7)
                      : const Color(0xFF16A34A),
                ),
              ),
            ),
          ],
        );
      }
    } else if (modelStatus.contains('Mengumpulkan')) {
      bodyContent = Text(
        'Peragakan gerakan isyarat...',
        style: TextStyle(
          fontSize: 15,
          fontWeight: FontWeight.w500,
          fontStyle: FontStyle.italic,
          color: isDark ? const Color(0xFF94A3B8) : const Color(0xFF64748B),
        ),
      );
    } else if (modelStatus.contains('Memproses')) {
      bodyContent = Text(
        'Menerjemahkan...',
        style: TextStyle(
          fontSize: 15,
          fontWeight: FontWeight.w500,
          fontStyle: FontStyle.italic,
          color: isDark ? const Color(0xFF94A3B8) : const Color(0xFF64748B),
        ),
      );
    } else {
      // Saat idle: Tampilan bersih "Belum ada hasil" tanpa teks "arahkan..."
      bodyContent = Text(
        'Belum ada hasil',
        style: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w500,
          color: isDark ? const Color(0xFF64748B) : const Color(0xFF94A3B8),
        ),
      );
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF1E293B) : Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: isDark ? 0.2 : 0.05),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Hasil Terjemahan',
                style: TextStyle(
                  fontWeight: FontWeight.w700,
                  fontSize: 15,
                  color: isDark ? Colors.white : const Color(0xFF1E293B),
                ),
              ),
              Flexible(
                child: FittedBox(
                  fit: BoxFit.scaleDown,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 4,
                    ),
                    decoration: BoxDecoration(
                      color: badgeBgColor,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      badgeText,
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: badgeTextColor,
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          bodyContent,
        ],
      ),
    );
  }

  Widget buildCameraPreview() {
    if (!hasCameraPermission) {
      return const Center(
        child: Text(
          'Izin kamera belum diberikan',
          style: TextStyle(color: Colors.black, fontWeight: FontWeight.w600),
        ),
      );
    }

    if (isSwitchingCamera) {
      return const Center(child: CircularProgressIndicator(color: mainBlue));
    }

    if (!isCameraOn) {
      return const Center(
        child: Icon(Icons.videocam_off, color: Colors.black, size: 82),
      );
    }

    if (!isCameraReady || cameraController == null) {
      return const Center(child: CircularProgressIndicator(color: mainBlue));
    }

    final previewSize = cameraController!.value.previewSize;
    if (previewSize == null) {
      return const Center(child: CircularProgressIndicator(color: mainBlue));
    }

    final isFrontCamera = _lensDirection == CameraLensDirection.front;

    Widget previewWidget = Stack(
      children: [
        Positioned.fill(child: CameraPreview(cameraController!)),
        if (detectedHandBox != null)
          Positioned(
            left: isFrontCamera
                ? (1.0 - detectedHandBox![2]) * previewSize.height
                : detectedHandBox![0] * previewSize.height,
            top: detectedHandBox![1] * previewSize.width,
            width:
                (detectedHandBox![2] - detectedHandBox![0]) *
                previewSize.height,
            height:
                (detectedHandBox![3] - detectedHandBox![1]) * previewSize.width,
            child: Container(
              decoration: BoxDecoration(
                border: Border.all(color: const Color(0xFF00FF00), width: 3.0),
                borderRadius: BorderRadius.circular(8),
              ),
            ),
          ),
      ],
    );

    // Do not mirror the front camera preview (mirroring disabled per request)
    /*
    if (isFrontCamera) {
      previewWidget = Transform.scale(
        scaleX: -1,
        child: previewWidget,
      );
    }
    */

    return ClipRect(
      child: FittedBox(
        fit: BoxFit.contain, // Mencegah zoom berlebihan, menampilkan area penuh
        child: SizedBox(
          width: previewSize.height,
          height: previewSize.width,
          child: previewWidget,
        ),
      ),
    );
  }
}
