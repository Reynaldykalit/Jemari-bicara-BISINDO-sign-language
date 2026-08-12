package com.example.jemari_bicara

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.graphics.Matrix
import android.os.Bundle
import android.os.Build
import android.app.ActivityManager
import android.content.Context
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import io.flutter.FlutterInjector
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import org.tensorflow.lite.Interpreter
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker

class MainActivity : FlutterActivity() {
    private var handLandmarker: HandLandmarker? = null
    private var yoloInterpreter: Interpreter? = null
    private var lstmInterpreter: Interpreter? = null
    private val channelName = "jemari_bicara/hand_landmarker"
    
    // === EXPERIMENT 1: YOLO BENCHMARK MODE ===
    private val ENABLE_YOLO_BENCHMARK = false

    // === Global Profiling Trackers ===
    private val sessionId = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date()) + "-" + UUID.randomUUID().toString().substring(0, 5)
    private var globalFrameCounter = 0L
    private var currentGestureId = ""
    private var gestureCounter = 0
    private var sessionInitLogged = false

    // Percentile trackers for current gesture
    private val latConversion = ArrayList<Double>()
    private val latYoloInference = ArrayList<Double>()
    private val latYoloPost = ArrayList<Double>()
    private val latMediaPipe = ArrayList<Double>()
    private val latTensorPrep = ArrayList<Double>()
    private val latTotalPrep = ArrayList<Double>()
    private val latEndToEnd = ArrayList<Double>()

    private var yoloAssetPath = "flutter_assets/assets/models/bisindo_hand_detector.tflite"
    private var landmarkerAssetPath = "flutter_assets/assets/models/hand_landmarker.task"
    private var lstmAssetPath = "flutter_assets/assets/models/classifier_final.tflite"

    private var yoloError: String? = null
    private var landmarkerError: String? = null
    private var lstmError: String? = null

    private var gestureBuffer = ArrayList<FloatArray>()
    private var lastLandmarksArray: FloatArray? = null

    // === Frame Loss & Latency Investigation Counters ===
    private var investigationFrameId = 0L
    private var sessionTotalCameraFrames = 0
    private var sessionProcessedByYolo = 0
    private var sessionWithLandmarks = 0
    private var sessionAddedToBuffer = 0
    private var sessionUsedByLstm = 0

    private var dropDroppedBeforeYolo = 0
    private var dropRejectedYolo = 0
    private var dropFailedMediaPipe = 0
    private var dropRejectedSegmentation = 0
    private var dropDiscardedMotionFilter = 0

    // Latency Accumulators for Session
    private var startCameraFramesCount = 0
    private var startSessionTimestamp = 0L
    private var accumCameraToYolo = 0.0
    private var accumYoloExec = 0.0
    private var accumYoloToMp = 0.0
    private var accumMpExec = 0.0
    private var accumMpToBuffer = 0.0
    private var accumTotalPrep = 0.0
    private var latencyCount = 0

    // === State Machine ===
    // Valid states: READY, GESTURE_COLLECTION, BUFFER_FROZEN, LSTM_INFERENCE, DISPLAY_RESULT, WAIT_NEXT_GESTURE
    private var status = "READY"
    private var waitNextGestureMissedFrames = 0

    // === Gesture Segmentation: Boundary-Only Counters ===
    private var missingFramesCount = 0
    private var lowMotionCounter = 0           // stable end detection
    private var lastBufferTime = 0L
    private var gestureStartTime = 0L
    private var hasMeaningfulMotion = false

    // === Phase 5 Validation Tracking ===
    private var peakMotion = 0.0f
    private var finalMotionValue = 0.0f
    private var freezeReason = ""
    private var cancelledFreezeAttempts = 0
    
    // === Phase 1: YOLO Coverage Audit ===
    private var phase1TotalCameraFrames = 0
    private var phase1TotalYoloDetections = 0
    private var phase1AcceptedDetections = 0
    private var phase1RejectedDetections = 0
    private var phase1FirstDetectionFrame = -1L
    private var phase1FirstAcceptedFrame = -1L
    private var phase1LastAcceptedFrame = -1L

    // === Phase 2: ROI Continuity Verification ===
    private var phase2LastRoiX = -1.0f
    private var phase2LastRoiY = -1.0f
    private var phase2LastRoiW = -1.0f
    private var phase2LastRoiH = -1.0f

    // === Phase 3: Gesture Acquisition Delay ===
    private var phase3MotionStartMs = -1L
    private var phase3FirstYoloDetectionMs = -1L
    private var phase3FirstAcceptedRoiMs = -1L
    private var phase3FirstLandmarkMs = -1L
    private var phase3FirstBufferedFrameMs = -1L
    
    // === Global Session Validation Metrics ===
    private var totalCompletedSessions = 0
    private var totalDiscardedSessions = 0
    private var completedInferenceSessions = 0
    private var highConfidencePredictions = 0

    // === Zero-Allocation Image Buffers ===
    private val yoloWidth = 640
    private val yoloHeight = 640
    private var yoloInputBuffer: ByteBuffer? = null
    private var yoloOutputArray = Array(1) { Array(5) { FloatArray(8400) } }
    
    private val mpWidth = 224
    private val mpHeight = 224
    private var mpBitmap: Bitmap? = null
    private var mpIntArray = IntArray(mpWidth * mpHeight)

    // === Inference State ===
    private var inferenceThread: Thread? = null
    private var currentPredictionConfidence = 0.0
    private var currentPredictionLabel = -1

    // Phase 1: Structured Profiling
    private var currentLstmTimeMs = 0.0
    private var currentMeanProbs = FloatArray(33) { 0f }
    private var currentTensorFirstFrame = emptyList<Float>()
    private var currentTensorLastFrame = emptyList<Float>()

    // Phase 2: Acquisition Analysis
    private var currentAcquisitionDelayMs = -1.0
    private var currentYoloCoverage = 0.0
    private var currentMotionYoloDelayMs = -1.0
    private var currentMotionRoiDelayMs = -1.0
    private var currentMotionMpDelayMs = -1.0
    private var currentMotionBufferDelayMs = -1.0

    // === Phase 3: YOLO ROI Reuse ===
    private var ENABLE_YOLO_SKIP = true
    private var yoloSkipLandmarkThreshold = 2
    private var yoloRerunMissThreshold = 2
    private var yoloRerunIntervalFrames = 15

    private var lastValidRoi: FloatArray? = null
    private var lastGlobalX = -1.0f
    private var lastGlobalY = -1.0f
    private var consecutiveLandmarkHits = 0
    private var consecutiveLandmarkMisses = 0
    private var yoloSkipCount = 0
    private var roiPaddingScale = 1.3f

    // === YOLO Tracker and Persistence Variables ===
    private var trackingState = "LOST"
    private var yoloLostCounter = 0
    private val maxLostFrames = 10
    private var lastSmoothedRoi = FloatArray(4) // [cx, cy, w, h]
    private var hasLastSmoothed = false
    private val smoothAlphaCoords = 0.70f
    private val smoothAlphaSize = 0.50f

    // === Pipeline Constants ===
    private val featureLength = 63
    private val WINDOW_SIZE = 30
    private val WINDOW_STEP = 5
    // Downsample indices: [0,3,6,9,12,15,18,21,24,27]
    private val DOWNSAMPLE_INDICES = intArrayOf(0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
    private val SEQUENCE_LENGTH = DOWNSAMPLE_INDICES.size  // 10
    private val MAX_BUFFER_FRAMES = 60             // safety cap
    private val MOTION_FILTER_THRESHOLD = 0.001f   // inference-time only
    private val LOW_MOTION_END_THRESHOLD = 0.004f  // stable end detection
    private val LOW_MOTION_CONSECUTIVE = 8          // ~0.8s stationary at 10fps
    private val MISSING_FRAME_TIMEOUT = 5           // ~0.5s lost hand

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        try {
            val loader = FlutterInjector.instance().flutterLoader()
            yoloAssetPath = loader.getLookupKeyForAsset("assets/models/bisindo_hand_detector.tflite")
            landmarkerAssetPath = loader.getLookupKeyForAsset("assets/models/hand_landmarker.task")
            lstmAssetPath = loader.getLookupKeyForAsset("assets/models/classifier_final.tflite")
        } catch (e: Exception) {
            e.printStackTrace()
        }

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "getNativeStatus" -> {
                        initModels()
                        val statusMap = mapOf(
                            "yoloReady" to (yoloInterpreter != null),
                            "yoloError" to yoloError,
                            "landmarkerReady" to (handLandmarker != null),
                            "landmarkerError" to landmarkerError,
                            "lstmReady" to (lstmInterpreter != null),
                            "lstmError" to lstmError
                        )
                        result.success(statusMap)
                    }
                    "clearBuffer" -> {
                        synchronized(gestureBuffer) {
                            val prevStatus = status
                            // RESET_SESSION → READY
                            gestureBuffer.clear()
                            lastValidRoi = null
                            lastLandmarksArray = null
                            lastGlobalX = -1.0f
                            lastGlobalY = -1.0f
                            lastBufferTime = 0L
                            missingFramesCount = 0
                            lowMotionCounter = 0
                            waitNextGestureMissedFrames = 0
                            trackingState = "LOST"
                            yoloLostCounter = 0
                            hasLastSmoothed = false
                            lastSmoothedRoi.fill(0.0f)
                            gestureStartTime = 0L
                            hasMeaningfulMotion = false
                            peakMotion = 0.0f
                            finalMotionValue = 0.0f
                            freezeReason = ""
                            cancelledFreezeAttempts = 0
                            currentPredictionLabel = -1
                            currentPredictionConfidence = 0.0
                            status = "READY"
                            
                            // Reset Phase 1-3 Investigation counters
                            phase1TotalCameraFrames = 0
                            phase1TotalYoloDetections = 0
                            phase1AcceptedDetections = 0
                            phase1RejectedDetections = 0
                            phase1FirstDetectionFrame = -1L
                            phase1FirstAcceptedFrame = -1L
                            phase1LastAcceptedFrame = -1L

                            phase2LastRoiX = -1.0f
                            phase2LastRoiY = -1.0f
                            phase2LastRoiW = -1.0f
                            phase2LastRoiH = -1.0f

                            phase3MotionStartMs = -1L
                            phase3FirstYoloDetectionMs = -1L
                            phase3FirstAcceptedRoiMs = -1L
                            phase3FirstLandmarkMs = -1L
                            phase3FirstBufferedFrameMs = -1L

                            // Reset investigation counters
                            investigationFrameId = 0L
                            sessionTotalCameraFrames = 0
                            sessionProcessedByYolo = 0
                            sessionWithLandmarks = 0
                            sessionAddedToBuffer = 0
                            sessionUsedByLstm = 0
                            dropDroppedBeforeYolo = 0
                            dropRejectedYolo = 0
                            dropFailedMediaPipe = 0
                            dropRejectedSegmentation = 0
                            dropDiscardedMotionFilter = 0
                            
                            startCameraFramesCount = 0
                            accumCameraToYolo = 0.0
                            accumYoloExec = 0.0
                            accumYoloToMp = 0.0
                            accumMpExec = 0.0
                            accumMpToBuffer = 0.0
                            accumTotalPrep = 0.0
                            latencyCount = 0

                            android.util.Log.d("PIPELINE_VALIDATION",
                                "[STATE] RESET_SESSION → READY | prevStatus=$prevStatus | ts=${System.currentTimeMillis()}")
                        }
                        result.success(true)
                    }
                    "processFrame" -> {
                        val width = call.argument<Int>("width") ?: 0
                        val height = call.argument<Int>("height") ?: 0
                        val rotation = call.argument<Int>("rotation") ?: 0
                        val planes = call.argument<List<Map<String, Any>>>("planes")
                        val confThresh = call.argument<Double>("confThresh")?.toFloat() ?: 0.17f
                        val startMotionThresh = call.argument<Double>("startMotionThresh")?.toFloat() ?: 0.012f
                        val endMotionThresh = call.argument<Double>("endMotionThresh")?.toFloat() ?: 0.006f
                        val minGestureLength = call.argument<Int>("minGestureLength") ?: 5
                        val maxGestureLength = call.argument<Int>("maxGestureLength") ?: 25
                        val cameraTime = call.argument<Long>("cameraTime") ?: 0L
                        val totalCameraFrames = call.argument<Int>("totalCameraFrames") ?: 0
                        val isFrontCamera = call.argument<Boolean>("isFrontCamera") ?: false

                        if (width == 0 || height == 0 || planes == null) {
                            result.success(null)
                            return@setMethodCallHandler
                        }
                        
                        ENABLE_YOLO_SKIP = call.argument<Boolean>("enableYoloSkip") ?: true
                        yoloSkipLandmarkThreshold = call.argument<Int>("yoloSkipLandmarkThreshold") ?: 2
                        yoloRerunMissThreshold = call.argument<Int>("yoloRerunMissThreshold") ?: 2
                        yoloRerunIntervalFrames = call.argument<Int>("yoloRerunIntervalFrames") ?: 15
                        roiPaddingScale = call.argument<Double>("roiPaddingScale")?.toFloat() ?: 1.3f

                        val threadStart = System.nanoTime()
                        Thread {
                            val threadCreationTimeMs = (System.nanoTime() - threadStart) / 1_000_000.0
                            try {
                                initModels()
                                if (yoloInterpreter == null || handLandmarker == null || lstmInterpreter == null) {
                                    val errMap = mapOf(
                                        "status" to "error",
                                        "message" to "Models not loaded. YOLO: ${yoloInterpreter != null}, MP: ${handLandmarker != null}, LSTM: ${lstmInterpreter != null}"
                                    )
                                    runOnUiThread { result.success(errMap) }
                                    return@Thread
                                }

                                val prediction = processFrame(width, height, rotation, planes, confThresh, startMotionThresh, endMotionThresh, minGestureLength, maxGestureLength, cameraTime, totalCameraFrames, threadCreationTimeMs, isFrontCamera)
                                runOnUiThread { result.success(prediction) }
                            } catch (e: Exception) {
                                e.printStackTrace()
                                val errMap = mapOf(
                                    "status" to "error",
                                    "message" to e.toString()
                                )
                                runOnUiThread { result.success(errMap) }
                            }
                        }.start()
                    }
                    else -> {
                        result.notImplemented()
                    }
                }
            }
    }

    private fun initModels() {
        if (yoloInterpreter == null) {
            try {
                val fileDescriptor = assets.openFd(yoloAssetPath)
                val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
                val fileChannel = inputStream.channel
                val startOffset = fileDescriptor.startOffset
                val declaredLength = fileDescriptor.declaredLength
                val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

                val options = Interpreter.Options()
                
                try {
                    val compatList = org.tensorflow.lite.gpu.CompatibilityList()
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        val delegateOptions = compatList.bestOptionsForThisDevice
                        options.addDelegate(org.tensorflow.lite.gpu.GpuDelegate(delegateOptions))
                        android.util.Log.d("PIPELINE_VALIDATION", "YOLO initialized with GPU Delegate")
                    } else {
                        options.setNumThreads(4)
                        options.setUseXNNPACK(true)
                        android.util.Log.d("PIPELINE_VALIDATION", "YOLO initialized with CPU/XNNPACK (GPU unsupported)")
                    }
                } catch (e: Exception) {
                    android.util.Log.w("PIPELINE_VALIDATION", "Failed to initialize GPU Delegate, falling back to CPU", e)
                    options.setNumThreads(4)
                    options.setUseXNNPACK(true)
                }
                
                yoloInterpreter = Interpreter(modelBuffer, options)
                yoloError = null
            } catch (e: Exception) {
                e.printStackTrace()
                yoloError = e.toString()
            }
        }

        if (handLandmarker == null) {
            try {
                val baseOptions = BaseOptions.builder()
                    .setModelAssetPath(landmarkerAssetPath)
                    .build()
                val options = HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.IMAGE)
                    .setNumHands(1)
                    .setMinHandDetectionConfidence(0.5f)
                    .build()
                handLandmarker = HandLandmarker.createFromOptions(this, options)
                landmarkerError = null
            } catch (e: Exception) {
                e.printStackTrace()
                landmarkerError = e.toString()
            }
        }

        if (lstmInterpreter == null) {
            try {
                val fileDescriptor = assets.openFd(lstmAssetPath)
                val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
                val fileChannel = inputStream.channel
                val startOffset = fileDescriptor.startOffset
                val declaredLength = fileDescriptor.declaredLength
                val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

                val options = Interpreter.Options()
                options.setNumThreads(2)
                lstmInterpreter = Interpreter(modelBuffer, options)
                lstmError = null
            } catch (e: Exception) {
                e.printStackTrace()
                lstmError = e.toString()
            }
        }

        if (!sessionInitLogged && yoloInterpreter != null && lstmInterpreter != null) {
            sessionInitLogged = true
            
            val actManager = this.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val memInfo = ActivityManager.MemoryInfo()
            actManager.getMemoryInfo(memInfo)
            
            android.util.Log.d("PIPELINE_VALIDATION", "===========================================")
            android.util.Log.d("PIPELINE_VALIDATION", "SESSION INITIALIZED: $sessionId")
            android.util.Log.d("PIPELINE_VALIDATION", "Device: ${Build.MANUFACTURER} ${Build.MODEL} (Android ${Build.VERSION.RELEASE}, API ${Build.VERSION.SDK_INT})")
            android.util.Log.d("PIPELINE_VALIDATION", "CPU ABI: ${Build.SUPPORTED_ABIS.joinToString(", ")} | Cores: ${Runtime.getRuntime().availableProcessors()}")
            android.util.Log.d("PIPELINE_VALIDATION", "Total RAM: ${memInfo.totalMem / 1024 / 1024} MB | Avail RAM: ${memInfo.availMem / 1024 / 1024} MB")
            
            android.util.Log.d("PIPELINE_VALIDATION", "--- YOLO Model ---")
            val yIn = yoloInterpreter?.getInputTensor(0)
            val yOut = yoloInterpreter?.getOutputTensor(0)
            android.util.Log.d("PIPELINE_VALIDATION", "Input Shape: ${yIn?.shape()?.contentToString()} | Type: ${yIn?.dataType()}")
            android.util.Log.d("PIPELINE_VALIDATION", "Output Shape: ${yOut?.shape()?.contentToString()} | Type: ${yOut?.dataType()}")
            
            android.util.Log.d("PIPELINE_VALIDATION", "--- LSTM Model ---")
            val lIn = lstmInterpreter?.getInputTensor(0)
            val lOut = lstmInterpreter?.getOutputTensor(0)
            android.util.Log.d("PIPELINE_VALIDATION", "Input Shape: ${lIn?.shape()?.contentToString()} | Type: ${lIn?.dataType()}")
            android.util.Log.d("PIPELINE_VALIDATION", "Output Shape: ${lOut?.shape()?.contentToString()} | Type: ${lOut?.dataType()}")
            android.util.Log.d("PIPELINE_VALIDATION", "===========================================")
        }
    }

    private fun processFrame(
        width: Int,
        height: Int,
        rotation: Int,
        planes: List<Map<String, Any>>,
        confThresh: Float,
        startMotionThresh: Float,
        endMotionThresh: Float,
        minGestureLength: Int,
        maxGestureLength: Int,
        cameraTime: Long,
        totalCameraFrames: Int,
        threadCreationTimeMs: Double,
        isFrontCamera: Boolean
    ): Map<String, Any>? {
        val totalStart = System.nanoTime()
        val timestamp = System.currentTimeMillis()

        globalFrameCounter++
        val frameId = globalFrameCounter

        synchronized(gestureBuffer) {
            if (status == "GESTURE_COLLECTION") {
                phase1TotalCameraFrames++
            }
        }

        synchronized(gestureBuffer) {
            investigationFrameId++
            if (status == "GESTURE_COLLECTION") {
                sessionProcessedByYolo++
            }
        }

        // Thread Utilization Logging
        val currentThread = Thread.currentThread()
        val jniLatencyMs = if (cameraTime > 0) (timestamp - cameraTime).toDouble() else -1.0
        android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Stage 1] Camera Frame Received | Thread=${currentThread.name} (ID=${currentThread.id}) | CameraTime=$cameraTime | JNI Queue Wait=$jniLatencyMs ms")

        // 1. Stage A - Camera Image Conversion
        val convStart = System.nanoTime()
        
        if (yoloInputBuffer == null) {
            yoloInputBuffer = ByteBuffer.allocateDirect(1 * yoloWidth * yoloHeight * 3 * 4)
            yoloInputBuffer!!.order(ByteOrder.nativeOrder())
        }
        if (mpBitmap == null) {
            mpBitmap = Bitmap.createBitmap(mpWidth, mpHeight, Bitmap.Config.ARGB_8888)
        }
        
        val extStart = System.nanoTime()
        val extTime = (System.nanoTime() - extStart) / 1_000_000.0
        
        val jpegTime = 0.0
        val rotTimeMs = 0.0
        val scaleTime = 0.0
        
        var tensorTime = 0.0
        var convTimeMs = 0.0
        var inferTime = 0.0
        var nmsTime = 0.0
        var yoloTimeMs = 0.0
        var yoloStart = 0L
        var yoloEnd = 0L
        
        var rawYoloDetectionsCount = 0
        var maxConf = 0.0f
        var bestX = 0.0f
        var bestY = 0.0f
        var bestW = 0.0f
        var bestH = 0.0f
        
        val shouldRunYolo = when {
            !ENABLE_YOLO_SKIP -> true
            lastValidRoi == null -> true
            consecutiveLandmarkMisses >= yoloRerunMissThreshold -> true
            yoloSkipCount >= yoloRerunIntervalFrames -> true
            consecutiveLandmarkHits < yoloSkipLandmarkThreshold -> true
            else -> false
        }
        
        var yoloSkippedThisFrame = false

        if (shouldRunYolo) {
            val tensorStart = System.nanoTime()
            convertYuvToFloatBuffer(planes, width, height, rotation, yoloWidth, yoloHeight, yoloInputBuffer!!)
            tensorTime = (System.nanoTime() - tensorStart) / 1_000_000.0

            convTimeMs = (System.nanoTime() - convStart) / 1_000_000.0
            if (status == "GESTURE_COLLECTION") { latConversion.add(convTimeMs); latTensorPrep.add(tensorTime) }
            
            android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Stage A] Image Conversion: Total=$convTimeMs ms | ext=$extTime | jpeg=$jpegTime | rot=$rotTimeMs | scale=$scaleTime | tensor=$tensorTime")

            yoloStart = System.nanoTime()
            
            val inferStart = System.nanoTime()
            yoloInterpreter?.run(yoloInputBuffer!!, yoloOutputArray)
            inferTime = (System.nanoTime() - inferStart) / 1_000_000.0

            val nmsStart = System.nanoTime()
            var bestScore = -Float.MAX_VALUE
            for (i in 0 until 8400) {
                val conf = yoloOutputArray[0][4][i]
                if (conf > 0.05f) { // Only consider plausible boxes
                    rawYoloDetectionsCount++
                    val x = yoloOutputArray[0][0][i]
                    val y = yoloOutputArray[0][1][i]
                    
                    var score = conf
                    if (lastValidRoi != null) {
                        // Spatial Continuity: strongly penalize jumping to a new hand far away
                        val dx = x - lastValidRoi!![0]
                        val dy = y - lastValidRoi!![1]
                        score -= (dx * dx + dy * dy) * 5.0f
                    } else {
                        // Hand Preference: Prioritize the right hand
                        // In a mirrored front-camera preview, the user's right hand appears on the right side of the screen (x > 0.5).
                        val prefX = if (isFrontCamera) x else (1.0f - x)
                        score += prefX * 0.15f
                    }

                    if (score > bestScore) {
                        bestScore = score
                        maxConf = conf
                        bestX = x
                        bestY = y
                        bestW = yoloOutputArray[0][2][i]
                        bestH = yoloOutputArray[0][3][i]
                    }
                }
            }
            nmsTime = (System.nanoTime() - nmsStart) / 1_000_000.0
            yoloEnd = System.nanoTime()
            yoloTimeMs = (yoloEnd - yoloStart) / 1_000_000.0

            if (globalFrameCounter == 1L) {
                android.util.Log.d("PIPELINE_VALIDATION", "[Phase 2] YOLO Warmup: First frame took $yoloTimeMs ms")
            }
            
            yoloSkipCount = 0
            // Removed unconditional lastValidRoi update to prevent ghost-hand teleportation
        } else {
            yoloSkippedThisFrame = true
            yoloSkipCount++
            bestX = lastValidRoi!![0]
            bestY = lastValidRoi!![1]
            bestW = lastValidRoi!![2]
            bestH = lastValidRoi!![3]
            maxConf = 1.0f // Dummy confidence
            
            android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Phase 3] YOLO Skipped. Reusing ROI: Box=[$bestX, $bestY, $bestW, $bestH]")
        }

        var yoloSuccess = maxConf >= confThresh
        val isNoDetection = maxConf < 0.01f
        var iou = 0.0f

        if (shouldRunYolo) {
            if (yoloSuccess) {
                // ROI Stabilization: Clamp box sizes to a minimum to prevent shrinking to 0
                bestW = bestW.coerceAtLeast(0.15f)
                bestH = bestH.coerceAtLeast(0.15f)

                val currentYoloBox = floatArrayOf(bestX, bestY, bestW, bestH)
                if (hasLastSmoothed) {
                    iou = calculateIoU(currentYoloBox, lastSmoothedRoi)
                    
                    if (iou >= 0.30f) {
                        // Temporal Smoothing using Exponential Moving Average (EMA)
                        val alphaX = smoothAlphaCoords
                        val alphaS = smoothAlphaSize
                        lastSmoothedRoi[0] = alphaX * bestX + (1 - alphaX) * lastSmoothedRoi[0]
                        lastSmoothedRoi[1] = alphaX * bestY + (1 - alphaX) * lastSmoothedRoi[1]
                        lastSmoothedRoi[2] = alphaS * bestW + (1 - alphaS) * lastSmoothedRoi[2]
                        lastSmoothedRoi[3] = alphaS * bestH + (1 - alphaS) * lastSmoothedRoi[3]
                        
                        bestX = lastSmoothedRoi[0]
                        bestY = lastSmoothedRoi[1]
                        bestW = lastSmoothedRoi[2]
                        bestH = lastSmoothedRoi[3]
                        
                        yoloLostCounter = 0
                        trackingState = "TRACKING"
                    } else if (status == "GESTURE_COLLECTION") {
                        // Fast movement during gesture: jump ROI instantly to new box to follow hand,
                        // but do NOT clear the gesture buffer!
                        lastSmoothedRoi[0] = bestX
                        lastSmoothedRoi[1] = bestY
                        lastSmoothedRoi[2] = bestW
                        lastSmoothedRoi[3] = bestH
                        
                        bestX = lastSmoothedRoi[0]
                        bestY = lastSmoothedRoi[1]
                        bestW = lastSmoothedRoi[2]
                        bestH = lastSmoothedRoi[3]
                        
                        yoloLostCounter = 0
                        trackingState = "FAST_TRACK"
                        android.util.Log.d("YOLO_TRACKER", "Fast Track (Low IoU $iou) during gesture collection. Jump ROI without clearing buffer.")
                    } else {
                        // Detection Recovery: low overlap indicates a new hand or a jump
                        // Reset tracking state & start fresh with the new YOLO box
                        lastSmoothedRoi[0] = bestX
                        lastSmoothedRoi[1] = bestY
                        lastSmoothedRoi[2] = bestW
                        lastSmoothedRoi[3] = bestH
                        yoloLostCounter = 0
                        trackingState = "NEW_TRACK"
                        
                        synchronized(gestureBuffer) {
                            gestureBuffer.clear()
                            consecutiveLandmarkHits = 0
                            consecutiveLandmarkMisses = 0
                        }
                        android.util.Log.d("YOLO_TRACKER", "Detection Recovery: Low IoU ($iou). Reset tracking to new box.")
                    }
                } else {
                    // First time detection
                    lastSmoothedRoi[0] = bestX
                    lastSmoothedRoi[1] = bestY
                    lastSmoothedRoi[2] = bestW
                    lastSmoothedRoi[3] = bestH
                    hasLastSmoothed = true
                    yoloLostCounter = 0
                    trackingState = "TRACKING"
                }
            } else {
                // YOLO did not detect any hand above threshold
                yoloLostCounter++
                if (hasLastSmoothed && yoloLostCounter <= maxLostFrames) {
                    // Bounding Box Persistence: reuse last smoothed ROI as temporary detection box
                    bestX = lastSmoothedRoi[0]
                    bestY = lastSmoothedRoi[1]
                    bestW = lastSmoothedRoi[2]
                    bestH = lastSmoothedRoi[3]
                    
                    maxConf = 1.0f // Force detection logic to proceed
                    yoloSuccess = true
                    trackingState = "PERSISTENT"
                    iou = 1.0f
                } else {
                    // Lost tracking completely
                    hasLastSmoothed = false
                    trackingState = "LOST"
                    yoloSuccess = false
                    iou = 0.0f
                }
            }
        } else {
            // YOLO was skipped, reuse lastValidRoi
            trackingState = "SKIPPED_REUSE"
            iou = 1.0f
        }

        // YOLO Tracker Logging
        android.util.Log.d("YOLO_TRACKER", java.lang.String.format(java.util.Locale.US,
            "Frame ID: %d | Detections: %d | Conf: %.4f | Box: [%.3f, %.3f, %.3f, %.3f] | IoU: %.4f | Lost Counter: %d | State: %s",
            frameId, rawYoloDetectionsCount, maxConf, bestX, bestY, bestW, bestH, iou, yoloLostCounter, trackingState
        ))

        synchronized(gestureBuffer) {
            if (status == "GESTURE_COLLECTION") {
                phase1TotalYoloDetections++
                
                if (phase1FirstDetectionFrame == -1L && !isNoDetection) {
                    phase1FirstDetectionFrame = frameId
                    phase3FirstYoloDetectionMs = timestamp
                }

                if (yoloSuccess) {
                    phase1AcceptedDetections++
                    if (phase1FirstAcceptedFrame == -1L) {
                        phase1FirstAcceptedFrame = frameId
                        phase3FirstAcceptedRoiMs = timestamp
                    }
                    phase1LastAcceptedFrame = frameId
                    
                    // Phase 2: ROI Continuity
                    if (phase2LastRoiW < 0) {
                        android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Phase 2] ROI APPEARED: X=$bestX Y=$bestY W=$bestW H=$bestH Conf=$maxConf")
                    } else {
                        val dx = Math.abs(bestX - phase2LastRoiX)
                        val dy = Math.abs(bestY - phase2LastRoiY)
                        val dw = Math.abs(bestW - phase2LastRoiW)
                        val dh = Math.abs(bestH - phase2LastRoiH)
                        if (dx > 0.15f || dy > 0.15f) {
                            android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Phase 2] ROI JUMPED: from ($phase2LastRoiX,$phase2LastRoiY) to ($bestX,$bestY) Conf=$maxConf")
                        }
                        if (dw > 0.15f || dh > 0.15f) {
                            android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Phase 2] ROI SIZE CHANGED: from ($phase2LastRoiW,$phase2LastRoiH) to ($bestW,$bestH) Conf=$maxConf")
                        }
                    }
                    phase2LastRoiX = bestX
                    phase2LastRoiY = bestY
                    phase2LastRoiW = bestW
                    phase2LastRoiH = bestH
                } else if (!isNoDetection) {
                    phase1RejectedDetections++
                    // Phase 2: Disappearance
                    if (phase2LastRoiW >= 0) {
                        android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Phase 2] ROI DISAPPEARED (Rejected/Lost)")
                        phase2LastRoiW = -1.0f
                    }
                } else {
                    // Phase 2: Disappearance (No detection)
                    if (phase2LastRoiW >= 0) {
                        android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Phase 2] ROI DISAPPEARED (No Candidate)")
                        phase2LastRoiW = -1.0f
                    }
                }
            }
        }

        if (status == "GESTURE_COLLECTION" && !yoloSkippedThisFrame) { latYoloInference.add(inferTime) }

        // Stage 2 Logging: YOLO Detection
        if (!yoloSkippedThisFrame) {
            android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Stage B] YOLO Inference: Pure Inference=$inferTime ms | NMS=$nmsTime ms | Hand=${if (yoloSuccess) "Yes" else "No"} | Conf=$maxConf | Box=[$bestX, $bestY, $bestW, $bestH]")
        }

        if (!yoloSuccess) {
            synchronized(gestureBuffer) {
                if (status == "GESTURE_COLLECTION" && !yoloSkippedThisFrame) {
                    dropRejectedYolo++
                }
            }
            android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Drop Reason] YOLO rejected (Conf $maxConf < $confThresh)")
        }

        if (ENABLE_YOLO_BENCHMARK) {
            val returnMap = HashMap<String, Any>()
            returnMap["status"] = "BENCHMARK_MODE"
            returnMap["windowSize"] = 0
            returnMap["label"] = -1
            returnMap["confidence"] = maxConf.toDouble()
            return returnMap
        }

        // 3. Stage C - YOLO Postprocessing (Crop & Resize ROI)
        val cropStart = System.nanoTime()
        val isCropped = yoloSuccess
        
        var px = bestX
        var py = bestY
        var pw = bestW
        var ph = bestH

        // Calculate original tight box (unpadded) for coordinate normalization later
        var tightPx = bestX
        var tightPy = bestY
        var tightPw = bestW
        var tightPh = bestH
        if (tightPw <= 1.0f && tightPh <= 1.0f) {
            tightPx *= yoloWidth
            tightPy *= yoloHeight
            tightPw *= yoloWidth
            tightPh *= yoloHeight
        }
        val tightX1 = tightPx - tightPw / 2.0f
        val tightY1 = tightPy - tightPh / 2.0f
        
        // Phase 5: ROI Expansion (Padding) to track fast movements
        if (isCropped) {
            pw = (pw * roiPaddingScale).coerceAtMost(1.0f)
            ph = (ph * roiPaddingScale).coerceAtMost(1.0f)
        }
        
        // If the model outputs normalized coordinates (0..1), scale them up to pixel space
        if (pw <= 1.0f && ph <= 1.0f) {
            px *= yoloWidth
            py *= yoloHeight
            pw *= yoloWidth
            ph *= yoloHeight
        }

        val x1 = if (isCropped) maxOf(0.0f, px - pw / 2.0f) else 0.0f
        val y1 = if (isCropped) maxOf(0.0f, py - ph / 2.0f) else 0.0f
        val x2 = if (isCropped) minOf(yoloWidth.toFloat() - 1f, px + pw / 2.0f) else 0.0f
        val y2 = if (isCropped) minOf(yoloHeight.toFloat() - 1f, py + ph / 2.0f) else 0.0f
        val cropWidth = (x2 - x1).toInt()
        val cropHeight = (y2 - y1).toInt()

        var handDetectedInMp = false
        var mpTimeMs = 0.0
        var detectTimeMs = 0.0
        var cropResizeTimeMs = 0.0
        var mpStart = 0L
        var mpEnd = 0L

        val landmarkerResult = if (isCropped && cropWidth > 0 && cropHeight > 0) {
            extractRoiToBitmap(planes, width, height, rotation, x1.toInt(), y1.toInt(), cropWidth, cropHeight, mpBitmap!!, mpIntArray)
            cropResizeTimeMs = (System.nanoTime() - cropStart) / 1_000_000.0
            val postProcessTotalMs = nmsTime + cropResizeTimeMs
            if (status == "GESTURE_COLLECTION") { latYoloPost.add(postProcessTotalMs) }
            android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Stage C] YOLO Postprocessing: Total=$postProcessTotalMs ms | NMS=$nmsTime ms | Crop/Resize=$cropResizeTimeMs ms")
            
            // 4. Stage D - MediaPipe Hand Landmarker
            mpStart = System.nanoTime()
            val mediaPipeImage = BitmapImageBuilder(mpBitmap!!).build()
            
            val detectStart = System.nanoTime()
            val lmr = handLandmarker?.detect(mediaPipeImage)
            detectTimeMs = (System.nanoTime() - detectStart) / 1_000_000.0
            
            mpEnd = System.nanoTime()
            mpTimeMs = (mpEnd - mpStart) / 1_000_000.0

            lmr
        } else {
            val postProcessTotalMs = nmsTime
            if (status == "GESTURE_COLLECTION") { latYoloPost.add(postProcessTotalMs) }
            android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Stage C] YOLO Postprocessing: Total=$postProcessTotalMs ms | NMS=$nmsTime ms | Hand Not Detected")
            null
        }

        val allLandmarks = landmarkerResult?.landmarks()
        val handDetected = (allLandmarks != null && allLandmarks.isNotEmpty())

        // Phase 3: Tracking state update
        if (handDetected) {
            consecutiveLandmarkHits++
            consecutiveLandmarkMisses = 0
        } else {
            consecutiveLandmarkMisses++
            consecutiveLandmarkHits = 0
        }

        if (status == "GESTURE_COLLECTION") { latMediaPipe.add(mpTimeMs) }

        // Stage 3 Logging: MediaPipe
        android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Stage D] MediaPipe: Total=$mpTimeMs ms | Pure Detect=$detectTimeMs ms | Landmark Success=${if (handDetected) "Yes" else "No"}")

        if (yoloSuccess && !handDetected) {
            synchronized(gestureBuffer) {
                if (status == "GESTURE_COLLECTION") {
                    dropFailedMediaPipe++
                }
            }
            android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Drop Reason] MediaPipe detected no landmarks")
        }

        val normStart = System.nanoTime()
        val landmarksArray = FloatArray(featureLength) // elements default to 0.0f
        var landmarkMin = 0.0f
        var landmarkMax = 0.0f
        var landmarkAvg = 0.0f
        var hasNaN = false
        var hasInfinite = false

        if (handDetected) {
            // Hand Continuity: Select the hand closest to previous frame's wrist to prevent hand swapping
            var bestHandIdx = 0
            if (allLandmarks!!.size > 1 && lastLandmarksArray != null) {
                var minDistance = Float.MAX_VALUE
                val prevWristX = lastLandmarksArray!![0]
                val prevWristY = lastLandmarksArray!![1]
                
                for (hIdx in allLandmarks.indices) {
                    val candidateWrist = allLandmarks[hIdx][0]
                    val cLx = if (tightPw > 0) ((x1 + candidateWrist.x() * cropWidth) - tightX1) / tightPw else candidateWrist.x()
                    val cLy = if (tightPh > 0) ((y1 + candidateWrist.y() * cropHeight) - tightY1) / tightPh else candidateWrist.y()
                    
                    val dist = (cLx - prevWristX) * (cLx - prevWristX) + (cLy - prevWristY) * (cLy - prevWristY)
                    if (dist < minDistance) {
                        minDistance = dist
                        bestHandIdx = hIdx
                    }
                }
            }
            val landmarks = allLandmarks[bestHandIdx]
            var sum = 0.0f
            for (i in 0 until 21) {
                val lm = landmarks[i]
                val lx_raw = lm.x()
                val ly_raw = lm.y()
                val lz_raw = lm.z()

                // Convert from padded ROI (MediaPipe) back to TIGHT YOLO BOX coordinates (Matches app.py)
                val absX = x1 + lx_raw * cropWidth
                val absY = y1 + ly_raw * cropHeight
                
                val lx = if (tightPw > 0) (absX - tightX1) / tightPw else lx_raw
                val ly = if (tightPh > 0) (absY - tightY1) / tightPh else ly_raw
                val lz = lz_raw

                if (lx.isNaN() || ly.isNaN() || lz.isNaN()) hasNaN = true
                if (lx.isInfinite() || ly.isInfinite() || lz.isInfinite()) hasInfinite = true

                landmarksArray[i * 3] = lx
                landmarksArray[i * 3 + 1] = ly
                landmarksArray[i * 3 + 2] = lz

                sum += lx + ly + lz
                if (i == 0) {
                    landmarkMin = minOf(lx, ly, lz)
                    landmarkMax = maxOf(lx, ly, lz)
                } else {
                    landmarkMin = minOf(landmarkMin, lx, ly, lz)
                    landmarkMax = maxOf(landmarkMax, lx, ly, lz)
                }
            }
            
            // Phase 5B: Closed-loop Tracking
            // Calculate absolute center of the hand from MediaPipe landmarks
            var minAbsX = Float.MAX_VALUE
            var maxAbsX = -Float.MAX_VALUE
            var minAbsY = Float.MAX_VALUE
            var maxAbsY = -Float.MAX_VALUE
            for (i in 0 until 21) {
                val lm = landmarks[i]
                val absX = x1 + lm.x() * cropWidth
                val absY = y1 + lm.y() * cropHeight
                minAbsX = minOf(minAbsX, absX)
                maxAbsX = maxOf(maxAbsX, absX)
                minAbsY = minOf(minAbsY, absY)
                maxAbsY = maxOf(maxAbsY, absY)
            }
            val handCenterAbsX = (minAbsX + maxAbsX) / 2.0f
            val handCenterAbsY = (minAbsY + maxAbsY) / 2.0f
            
            // Phase 5B: Closed-loop Tracking Update (Ghost-Hand Immune)
            val mpCx = handCenterAbsX / yoloWidth
            val mpCy = handCenterAbsY / yoloHeight
            if (hasLastSmoothed) {
                val alphaX = smoothAlphaCoords
                lastSmoothedRoi[0] = alphaX * mpCx + (1 - alphaX) * lastSmoothedRoi[0]
                lastSmoothedRoi[1] = alphaX * mpCy + (1 - alphaX) * lastSmoothedRoi[1]
                
                // Stable Box Size: Only update width/height when YOLO runs, keeping box size completely stable during tracking
                if (!yoloSkippedThisFrame && yoloSuccess && trackingState != "PERSISTENT") {
                    val alphaS = smoothAlphaSize
                    lastSmoothedRoi[2] = alphaS * bestW + (1 - alphaS) * lastSmoothedRoi[2]
                    lastSmoothedRoi[3] = alphaS * bestH + (1 - alphaS) * lastSmoothedRoi[3]
                }
            } else {
                lastSmoothedRoi[0] = mpCx
                lastSmoothedRoi[1] = mpCy
                lastSmoothedRoi[2] = bestW
                lastSmoothedRoi[3] = bestH
                hasLastSmoothed = true
            }
            
            // Sync lastValidRoi with lastSmoothedRoi so YOLO skip logic knows where to crop
            if (lastValidRoi == null) {
                lastValidRoi = lastSmoothedRoi.clone()
            } else {
                lastValidRoi!![0] = lastSmoothedRoi[0]
                lastValidRoi!![1] = lastSmoothedRoi[1]
                lastValidRoi!![2] = lastSmoothedRoi[2]
                lastValidRoi!![3] = lastSmoothedRoi[3]
            }

            // Recover tracking if landmarks detected
            yoloLostCounter = 0
            if (trackingState == "PERSISTENT") {
                trackingState = "RECOVERED"
            }
            landmarkAvg = sum / featureLength
            synchronized(gestureBuffer) {
                if (status == "GESTURE_COLLECTION") {
                    sessionWithLandmarks++
                    if (phase3FirstLandmarkMs == -1L) {
                        phase3FirstLandmarkMs = timestamp
                    }
                }
            }
        }
        val normTimeMs = (System.nanoTime() - normStart) / 1_000_000.0
        android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Stage D] Landmark Normalization: $normTimeMs ms")

        // Diagnostic variables for Flutter return map
        var windowSize = 0
        var labelIndex = currentPredictionLabel
        var confidenceValue = currentPredictionConfidence

        // Memory snapshot
        val freeMem = Runtime.getRuntime().freeMemory() / 1024 / 1024
        val totalMem = Runtime.getRuntime().totalMemory() / 1024 / 1024
        val usedMem = totalMem - freeMem
        android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Stage E] Memory: Used=$usedMem MB | Free=$freeMem MB | Total=$totalMem MB")

        val e2eTotalMs = (System.nanoTime() - totalStart) / 1_000_000.0
        if (status == "GESTURE_COLLECTION") {
            latTotalPrep.add(e2eTotalMs)
        }
        android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Stage G] Frame Preprocessing Total Latency: $e2eTotalMs ms")

        android.util.Log.d("PIPELINE", "[PIPELINE] FRAME_RECEIVED")
        synchronized(gestureBuffer) {
            val currentTimeMillis = System.currentTimeMillis()
            val t6 = System.nanoTime()

            // === CONCURRENCY LOCK ===
            // Block new data when inference is running or paused awaiting reset
            if (status == "WAIT_NEXT_GESTURE" || status == "DISPLAY_RESULT" ||
                status == "BUFFER_FROZEN" || status == "LSTM_INFERENCE") {
                if (status == "GESTURE_COLLECTION") {
                    dropDroppedBeforeYolo++
                }
                
                // Auto-reset from WAIT_NEXT_GESTURE state when hand is removed or moves again
                if (status == "WAIT_NEXT_GESTURE") {
                    if (!handDetected) {
                        waitNextGestureMissedFrames++
                        if (waitNextGestureMissedFrames >= MISSING_FRAME_TIMEOUT) {
                            status = "READY"
                            gestureBuffer.clear()
                            lastValidRoi = null
                            lastLandmarksArray = null
                            lastGlobalX = -1.0f
                            lastGlobalY = -1.0f
                            waitNextGestureMissedFrames = 0
                            lowMotionCounter = 0
                            gestureStartTime = 0L
                            hasMeaningfulMotion = false
                            android.util.Log.d("PIPELINE_VALIDATION", "[AUTO RESET] Hand removed. WAIT_NEXT_GESTURE -> READY")
                        }
                    } else {
                        waitNextGestureMissedFrames = 0
                        // Calculate motion to detect new movement
                        var frameMotion = 0.0f
                        if (lastLandmarksArray != null) {
                            var totalDist = 0.0f
                            for (i in 0 until 21) {
                                val dx = landmarksArray[i * 3]     - lastLandmarksArray!![i * 3]
                                val dy = landmarksArray[i * 3 + 1] - lastLandmarksArray!![i * 3 + 1]
                                val dz = landmarksArray[i * 3 + 2] - lastLandmarksArray!![i * 3 + 2]
                                totalDist += Math.sqrt((dx * dx + dy * dy + dz * dz).toDouble()).toFloat()
                            }
                            var fingerMotion = totalDist / 21f
                            var globalTranslation = 0.0f
                            if (lastValidRoi != null && lastGlobalX != -1.0f && lastGlobalY != -1.0f) {
                                val gdx = lastValidRoi!![0] - lastGlobalX
                                val gdy = lastValidRoi!![1] - lastGlobalY
                                globalTranslation = Math.sqrt((gdx * gdx + gdy * gdy).toDouble()).toFloat()
                            }
                            frameMotion = fingerMotion + globalTranslation
                        }
                        
                        // Update last-known landmarks and global position
                        lastLandmarksArray = landmarksArray.clone()
                        if (lastValidRoi != null) {
                            lastGlobalX = lastValidRoi!![0]
                            lastGlobalY = lastValidRoi!![1]
                        }
                        
                        // If significant motion is detected, auto-reset to READY
                        if (lastLandmarksArray != null && frameMotion > startMotionThresh) {
                            status = "READY"
                            gestureBuffer.clear()
                            waitNextGestureMissedFrames = 0
                            lowMotionCounter = 0
                            gestureStartTime = 0L
                            hasMeaningfulMotion = false
                            android.util.Log.d("PIPELINE_VALIDATION", "[AUTO RESET] Significant motion detected. WAIT_NEXT_GESTURE -> READY")
                        }
                    }
                }
                
                android.util.Log.d("PIPELINE_VALIDATION", "[STATE LOCK] Ignoring frame. Current status=$status")
                return@synchronized
            }

            // Limit sampling to ~10 FPS (100ms interval)
            if (currentTimeMillis - lastBufferTime < 100) {
                if (status == "GESTURE_COLLECTION") {
                    dropDroppedBeforeYolo++
                }
                return@synchronized
            }
            lastBufferTime = currentTimeMillis

            // === GESTURE SEGMENTATION (BOUNDARY ONLY — NO DATA MODIFICATION) ===
            var frameMotion = 0.0f

            if (handDetected) {
                // Compute frame motion for stable end detection
                if (lastLandmarksArray != null) {
                    var totalDist = 0.0f
                    for (i in 0 until 21) {
                        val dx = landmarksArray[i * 3]     - lastLandmarksArray!![i * 3]
                        val dy = landmarksArray[i * 3 + 1] - lastLandmarksArray!![i * 3 + 1]
                        val dz = landmarksArray[i * 3 + 2] - lastLandmarksArray!![i * 3 + 2]
                        totalDist += Math.sqrt((dx * dx + dy * dy + dz * dz).toDouble()).toFloat()
                    }
                    var fingerMotion = totalDist / 21f
                    
                    // Add global hand translation to prevent premature stopping
                    var globalTranslation = 0.0f
                    if (lastValidRoi != null && lastGlobalX != -1.0f && lastGlobalY != -1.0f) {
                        val gdx = lastValidRoi!![0] - lastGlobalX
                        val gdy = lastValidRoi!![1] - lastGlobalY
                        globalTranslation = Math.sqrt((gdx * gdx + gdy * gdy).toDouble()).toFloat()
                    }
                    frameMotion = fingerMotion + globalTranslation
                }
                
                // Update last-known landmarks and global position
                lastLandmarksArray = landmarksArray.clone()
                if (lastValidRoi != null) {
                    lastGlobalX = lastValidRoi!![0]
                    lastGlobalY = lastValidRoi!![1]
                }

                if (status == "GESTURE_COLLECTION") {
                    if (frameMotion > peakMotion) {
                        peakMotion = frameMotion
                    }
                    if (frameMotion > startMotionThresh) {
                        hasMeaningfulMotion = true
                        if (phase3MotionStartMs == -1L) {
                            phase3MotionStartMs = timestamp
                        }
                    }
                }

                // Stable end detection counter
                if (gestureBuffer.isNotEmpty()) {
                    if (frameMotion < LOW_MOTION_END_THRESHOLD) {
                        lowMotionCounter++
                    } else if (frameMotion > endMotionThresh) {
                        if (hasMeaningfulMotion && lowMotionCounter > 0) {
                            cancelledFreezeAttempts++
                        }
                        lowMotionCounter = 0
                    }
                    // Jitter margin: between LOW_MOTION_END_THRESHOLD and endMotionThresh, the counter pauses.
                }

                // Start new session if READY
                if (status == "READY") {
                    status = "GESTURE_COLLECTION"
                    gestureStartTime = currentTimeMillis
                    gestureCounter++
                    currentGestureId = String.format("%04d", gestureCounter)

                    // Clear percentile arrays
                    latConversion.clear()
                    latYoloInference.clear()
                    latYoloPost.clear()
                    latMediaPipe.clear()
                    latTensorPrep.clear()
                    latTotalPrep.clear()
                    latEndToEnd.clear()

                    // Reset all session counters at start
                    startCameraFramesCount = totalCameraFrames
                    startSessionTimestamp = currentTimeMillis
                    sessionTotalCameraFrames = 0
                    sessionProcessedByYolo = 1
                    sessionWithLandmarks = 1
                    sessionAddedToBuffer = 0
                    sessionUsedByLstm = 0
                    dropDroppedBeforeYolo = 0
                    dropRejectedYolo = 0
                    dropFailedMediaPipe = 0
                    dropRejectedSegmentation = 0
                    dropDiscardedMotionFilter = 0

                    accumCameraToYolo = 0.0
                    accumYoloExec = 0.0
                    accumYoloToMp = 0.0
                    accumMpExec = 0.0
                    accumMpToBuffer = 0.0
                    accumTotalPrep = 0.0
                    latencyCount = 0
                    hasMeaningfulMotion = false
                    peakMotion = 0.0f
                    finalMotionValue = 0.0f
                    freezeReason = ""
                    cancelledFreezeAttempts = 0

                    android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Stage 0] READY → GESTURE_COLLECTION")
                }

                // Collect frame into bounded buffer (boundary only, strict chronological append)
                if (gestureBuffer.size < MAX_BUFFER_FRAMES) {
                    gestureBuffer.add(landmarksArray.clone())
                    sessionAddedToBuffer++
                    if (phase3FirstBufferedFrameMs == -1L) {
                        phase3FirstBufferedFrameMs = timestamp
                    }

                    // Calculate and accumulate latencies
                    val tFinalFrame = System.nanoTime()
                    val cameraToYolo = if (cameraTime > 0) (yoloStart - (cameraTime * 1_000_000L)) / 1_000_000.0 else 0.0
                    val yoloToMp = (mpStart - yoloEnd) / 1_000_000.0
                    val mpToBuffer = (t6 - mpEnd) / 1_000_000.0
                    val totalPrep = (tFinalFrame - totalStart) / 1_000_000.0

                    accumCameraToYolo += cameraToYolo
                    accumYoloExec += yoloTimeMs
                    accumYoloToMp += yoloToMp
                    accumMpExec += mpTimeMs
                    accumMpToBuffer += mpToBuffer
                    accumTotalPrep += totalPrep
                    latencyCount++

                    // Stage 4 Logging: Buffer Append
                    android.util.Log.d("PIPELINE_VALIDATION", "[Stage 4] Buffer Append | Index=${gestureBuffer.size - 1} | Size=${gestureBuffer.size} | ts=$currentTimeMillis")
                }
                missingFramesCount = 0

            } else {
                // Hand not detected this frame
                if (gestureBuffer.isNotEmpty()) {
                    missingFramesCount++
                }
            }

            // === GESTURE END DETECTION (STABLE) ===
            val isLostTimeout       = missingFramesCount >= MISSING_FRAME_TIMEOUT
            val isStationaryTimeout = gestureBuffer.size >= minGestureLength && lowMotionCounter >= LOW_MOTION_CONSECUTIVE
            val isMaxBuffer         = gestureBuffer.size >= MAX_BUFFER_FRAMES

            val endConditionMet = (isMaxBuffer || isLostTimeout || isStationaryTimeout)
            val gestureFinished = status == "GESTURE_COLLECTION" &&
                gestureBuffer.isNotEmpty() && endConditionMet

            // Stage 5 Logging: Gesture Segmentation
            android.util.Log.d("PIPELINE_VALIDATION",
                "[$sessionId][$currentGestureId][F-$frameId][$timestamp] [Stage 5] Segmentation | motion=$frameMotion | lowCnt=$lowMotionCounter | miss=$missingFramesCount | finished=$gestureFinished | meaningfulMotion=$hasMeaningfulMotion")

            if (!gestureFinished) {
                // Noise short-circuit: reset if hand lost before collecting enough
                if (isLostTimeout && gestureBuffer.isEmpty()) {
                    status = "READY"
                    missingFramesCount = 0
                    lowMotionCounter = 0
                    lastLandmarksArray = null
                    lastValidRoi = null
                    gestureStartTime = 0L
                    hasMeaningfulMotion = false
                }
                return@synchronized
            }

            // === DETERMINISTIC GESTURE SESSION FINALIZATION ===
            finalMotionValue = frameMotion
            freezeReason = when {
                isMaxBuffer -> "Max Buffer"
                isLostTimeout -> "Lost Tracking"
                isStationaryTimeout -> "Stationary"
                else -> "Unknown"
            }

            if (gestureBuffer.size < minGestureLength) {
                // Discard the session
                totalDiscardedSessions++
                android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId] [Stage 5] Discarding session (Too short: ${gestureBuffer.size} < $minGestureLength). Reason=$freezeReason")
                status = "READY"
                gestureBuffer.clear()
                missingFramesCount = 0
                lowMotionCounter = 0
                lastLandmarksArray = null
                lastValidRoi = null
                gestureStartTime = 0L
                hasMeaningfulMotion = false
                return@synchronized
            }

            totalCompletedSessions++

            // === BUFFER FREEZE — TRUE DEEP COPY ===
            // Element-by-element copy; camera thread must never touch this snapshot.
            val L = gestureBuffer.size
            val snapshot: Array<FloatArray> = Array(L) { i ->
                gestureBuffer[i].copyOf()   // true deep copy of each frame
            }
            status = "BUFFER_FROZEN"

            val gestureDurationVal = currentTimeMillis - gestureStartTime
            val averageBufferFps = if (gestureDurationVal > 0) (L * 1000.0 / gestureDurationVal) else 0.0

            android.util.Log.d("PIPELINE_VALIDATION", "===========================================")
            android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId] GESTURE SESSION LOG")
            android.util.Log.d("PIPELINE_VALIDATION", "--- Gesture Session ---")
            android.util.Log.d("PIPELINE_VALIDATION", "Gesture ID       : $currentGestureId")
            android.util.Log.d("PIPELINE_VALIDATION", "Start Time       : $gestureStartTime")
            android.util.Log.d("PIPELINE_VALIDATION", "End Time         : $currentTimeMillis")
            android.util.Log.d("PIPELINE_VALIDATION", "Duration         : $gestureDurationVal ms")
            android.util.Log.d("PIPELINE_VALIDATION", "Total Frames     : $L")
            android.util.Log.d("PIPELINE_VALIDATION", "Buffer Avg FPS   : ${String.format(Locale.US, "%.2f", averageBufferFps)}")
            android.util.Log.d("PIPELINE_VALIDATION", "Peak Motion      : ${String.format(Locale.US, "%.4f", peakMotion)}")
            android.util.Log.d("PIPELINE_VALIDATION", "Final Motion     : ${String.format(Locale.US, "%.4f", finalMotionValue)}")
            android.util.Log.d("PIPELINE_VALIDATION", "Freeze Reason    : $freezeReason")
            android.util.Log.d("PIPELINE_VALIDATION", "--- Motion Stabilization ---")
            android.util.Log.d("PIPELINE_VALIDATION", "Current Motion   : ${String.format(Locale.US, "%.4f", frameMotion)}")
            android.util.Log.d("PIPELINE_VALIDATION", "lowMotionCounter : $lowMotionCounter")
            android.util.Log.d("PIPELINE_VALIDATION", "Threshold Value  : $LOW_MOTION_END_THRESHOLD")
            android.util.Log.d("PIPELINE_VALIDATION", "Cancelled Freezes: $cancelledFreezeAttempts")
            android.util.Log.d("PIPELINE_VALIDATION", "===========================================")
            android.util.Log.d("PIPELINE_VALIDATION", "--- Phase 1: YOLO Detection Coverage Audit ---")
            android.util.Log.d("PIPELINE_VALIDATION", "Total Camera Frames : $phase1TotalCameraFrames")
            android.util.Log.d("PIPELINE_VALIDATION", "Total YOLO Detects  : $phase1TotalYoloDetections")
            android.util.Log.d("PIPELINE_VALIDATION", "Accepted Detections : $phase1AcceptedDetections")
            android.util.Log.d("PIPELINE_VALIDATION", "Rejected Detections : $phase1RejectedDetections")
            android.util.Log.d("PIPELINE_VALIDATION", "First Detect Frame  : $phase1FirstDetectionFrame")
            android.util.Log.d("PIPELINE_VALIDATION", "First Accepted Frame: $phase1FirstAcceptedFrame")
            android.util.Log.d("PIPELINE_VALIDATION", "Last Accepted Frame : $phase1LastAcceptedFrame")
            val coveragePercent = if (phase1TotalCameraFrames > 0) (phase1AcceptedDetections.toFloat() / phase1TotalCameraFrames * 100) else 0.0f
            currentYoloCoverage = coveragePercent.toDouble()
            android.util.Log.d("PIPELINE_VALIDATION", "Overall Coverage    : ${String.format(Locale.US, "%.2f", coveragePercent)}%")

            android.util.Log.d("PIPELINE_VALIDATION", "--- Phase 3: Gesture Acquisition Delay Analysis ---")
            val motionYoloDelay = if (phase3MotionStartMs > 0 && phase3FirstYoloDetectionMs > 0) phase3FirstYoloDetectionMs - phase3MotionStartMs else -1
            val motionRoiDelay = if (phase3MotionStartMs > 0 && phase3FirstAcceptedRoiMs > 0) phase3FirstAcceptedRoiMs - phase3MotionStartMs else -1
            val motionMpDelay = if (phase3MotionStartMs > 0 && phase3FirstLandmarkMs > 0) phase3FirstLandmarkMs - phase3MotionStartMs else -1
            val motionBufferDelay = if (phase3MotionStartMs > 0 && phase3FirstBufferedFrameMs > 0) phase3FirstBufferedFrameMs - phase3MotionStartMs else -1
            
            currentMotionYoloDelayMs = motionYoloDelay.toDouble()
            currentMotionRoiDelayMs = motionRoiDelay.toDouble()
            currentMotionMpDelayMs = motionMpDelay.toDouble()
            currentMotionBufferDelayMs = motionBufferDelay.toDouble()
            currentAcquisitionDelayMs = motionBufferDelay.toDouble()
            
            android.util.Log.d("PIPELINE_VALIDATION", "Motion -> YOLO Delay: $motionYoloDelay ms")
            android.util.Log.d("PIPELINE_VALIDATION", "Motion -> ROI Delay : $motionRoiDelay ms")
            android.util.Log.d("PIPELINE_VALIDATION", "Motion -> MP Delay  : $motionMpDelay ms")
            android.util.Log.d("PIPELINE_VALIDATION", "Motion -> Buf Delay : $motionBufferDelay ms")
            android.util.Log.d("PIPELINE_VALIDATION", "===========================================")

            // Launch inference on background thread
            inferenceThread = Thread {

                // === SLIDING WINDOW CONSTRUCTION (Training-Aligned) ===
                val windows = ArrayList<Array<FloatArray>>()

                if (L < WINDOW_SIZE) {
                    // Phase 5: Temporal Stretching (Linear Interpolation) for smooth velocity
                    val padded = Array(WINDOW_SIZE) { i ->
                        val exactIdx = if (WINDOW_SIZE > 1) i.toFloat() / (WINDOW_SIZE - 1) * (L - 1) else 0f
                        val idx0 = exactIdx.toInt().coerceAtMost(L - 1)
                        val idx1 = (idx0 + 1).coerceAtMost(L - 1)
                        val weight = exactIdx - idx0
                        
                        val interp = FloatArray(featureLength)
                        for (j in 0 until featureLength) {
                            interp[j] = snapshot[idx0][j] * (1.0f - weight) + snapshot[idx1][j] * weight
                        }
                        interp
                    }
                    windows.add(padded)
                    android.util.Log.d("PIPELINE_VALIDATION",
                        "[Stage 6] Sequence < 30 | temporally stretched (linear) from $L → $WINDOW_SIZE | windows=1")
                } else {
                    // Generate overlapping sliding windows
                    var start = 0
                    while (start + WINDOW_SIZE <= L) {
                        val w = Array(WINDOW_SIZE) { i -> snapshot[start + i].copyOf() }
                        windows.add(w)
                        start += WINDOW_STEP
                    }
                    android.util.Log.d("PIPELINE_VALIDATION",
                        "[Stage 6] Sliding windows | L=$L | size=$WINDOW_SIZE | step=$WINDOW_STEP | count=${windows.size}")
                }

                // === MOTION FILTERING (inference-time only) ===
                val validWindows = ArrayList<Array<FloatArray>>()
                val motionScores  = ArrayList<Float>()
                val windowIndices = ArrayList<Int>()

                for ((winIdx, w) in windows.withIndex()) {
                    val score = calculateWindowMotion(w)
                    motionScores.add(score)
                    if (score >= MOTION_FILTER_THRESHOLD) {
                        validWindows.add(w)
                        windowIndices.add(winIdx)
                    } else {
                        dropDiscardedMotionFilter++
                    }
                }

                android.util.Log.d("PIPELINE_VALIDATION",
                    "[Stage 6] Motion scores=${motionScores} | threshold=$MOTION_FILTER_THRESHOLD | " +
                    "valid=${validWindows.size} | rejected=${windows.size - validWindows.size}")

                if (validWindows.isEmpty()) {
                    // No moving window → gesture not recognized
                    currentPredictionLabel = -1
                    currentPredictionConfidence = 0.0
                    android.util.Log.d("PIPELINE_VALIDATION",
                        "[$sessionId][$currentGestureId] [Stage 6] ALL windows rejected by motion filter → labelIndex=-1")

                    // Log the drop summary even if rejected early
                    printDropSummary(totalCameraFrames, L, emptyList(), windows.size, validWindows.size, gestureDurationVal, -1, 0.0)
                    printLatencySummary(0.0)
                    status = "DISPLAY_RESULT"
                    return@Thread
                }

                // Calculate unique indices used by LSTM
                val uniqueFrameIndices = HashSet<Int>()
                for (winIdx in windowIndices) {
                    if (L < WINDOW_SIZE) {
                        for (idx in DOWNSAMPLE_INDICES) {
                            if (idx < L) {
                                uniqueFrameIndices.add(idx)
                            }
                        }
                    } else {
                        val start = winIdx * WINDOW_STEP
                        for (idx in DOWNSAMPLE_INDICES) {
                            uniqueFrameIndices.add(start + idx)
                        }
                    }
                }
                sessionUsedByLstm = uniqueFrameIndices.size

                // === MULTI-WINDOW INFERENCE + MEAN PROBABILITY AGGREGATION ===
                val meanProbs = FloatArray(33) { 0f }
                val lstmStart = System.nanoTime()

                for ((winIdx, w) in validWindows.withIndex()) {

                    // Downsample 30 → 10
                    val downsampled = Array(SEQUENCE_LENGTH) { i -> w[DOWNSAMPLE_INDICES[i]].copyOf() }

                    // Construct interpreter input
                    val inputTensor = Array(1) { Array(SEQUENCE_LENGTH) { FloatArray(featureLength) } }
                    for (f in 0 until SEQUENCE_LENGTH) {
                        inputTensor[0][f] = downsampled[f]
                    }

                    // Stage 7 Logging: LSTM Input Preview
                    android.util.Log.d("PIPELINE_VALIDATION",
                        "[Stage 7] LSTM Input | Window $winIdx | Shape=[1,$SEQUENCE_LENGTH,$featureLength] | " +
                        "First frame sample=${downsampled[0].take(3).toTypedArray().contentToString()}... | " +
                        "Last frame sample=${downsampled[SEQUENCE_LENGTH-1].take(3).toTypedArray().contentToString()}...")

                    val outTensor = Array(1) { FloatArray(33) }
                    lstmInterpreter?.run(inputTensor, outTensor)

                    // Accumulate probabilities
                    for (c in 0 until 33) {
                        meanProbs[c] += outTensor[0][c]
                    }
                }

                val lstmEnd = System.nanoTime()
                val lstmTimeMs = (lstmEnd - lstmStart) / 1_000_000.0
                currentLstmTimeMs = lstmTimeMs

                // Mean probabilities
                for (c in 0 until 33) { meanProbs[c] /= validWindows.size }
                currentMeanProbs = meanProbs.copyOf()
                
                if (validWindows.isNotEmpty()) {
                    val lastWindow = validWindows.last()
                    currentTensorFirstFrame = lastWindow[DOWNSAMPLE_INDICES.first()].toList()
                    currentTensorLastFrame = lastWindow[DOWNSAMPLE_INDICES.last()].toList()
                }

                // Single argmax
                var bestIdx = 0; var bestConf = meanProbs[0]
                for (c in 1 until 33) {
                    if (meanProbs[c] > bestConf) { bestConf = meanProbs[c]; bestIdx = c }
                }

                if (bestConf >= confThresh) {
                    highConfidencePredictions++
                }

                currentPredictionLabel      = bestIdx
                currentPredictionConfidence = bestConf.toDouble()
                completedInferenceSessions++

                status = "DISPLAY_RESULT"
                android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId] [STATE] LSTM_INFERENCE → DISPLAY_RESULT")

                // Print the drop summary and latencies
                printDropSummary(totalCameraFrames, L, uniqueFrameIndices.toList(), windows.size, validWindows.size, gestureDurationVal, bestIdx, bestConf.toDouble())
                printLatencySummary(lstmTimeMs)
            }
            status = "LSTM_INFERENCE"
            inferenceThread?.start()
        }

        // Pass results appropriately
        windowSize = gestureBuffer.size

        if (status == "DISPLAY_RESULT" || status == "WAIT_NEXT_GESTURE") {
            labelIndex = currentPredictionLabel
            confidenceValue = currentPredictionConfidence
            // Auto-advance to WAIT_NEXT_GESTURE so Flutter knows to show result then reset
            if (status == "DISPLAY_RESULT") {
                status = "WAIT_NEXT_GESTURE"
                android.util.Log.d("PIPELINE_VALIDATION", "[STATE] DISPLAY_RESULT → WAIT_NEXT_GESTURE")
            }
        }

        val totalTimeMs = (System.nanoTime() - totalStart) / 1_000_000.0

        // Return a map containing all intermediate states & metrics
        val returnMap = HashMap<String, Any>()
        returnMap["status"] = status
        returnMap["timestamp"] = timestamp

        // Task 1: Camera
        returnMap["cameraWidth"] = width
        returnMap["cameraHeight"] = height
        returnMap["cameraRotation"] = rotation
        returnMap["cameraFacing"] = if (rotation == 270 || rotation == 90) "front" else "back"

        // Task 2: Image Conversion
        returnMap["conversionTimeMs"] = convTimeMs
        returnMap["yuvWidth"] = width
        returnMap["yuvHeight"] = height
        returnMap["pixelFormat"] = "NV21"

        // Task 3: Rotation
        returnMap["rotationTimeMs"] = rotTimeMs
        returnMap["originalWidth"] = width
        returnMap["originalHeight"] = height
        returnMap["rotatedWidth"] = if (rotation == 90 || rotation == 270) height else width
        returnMap["rotatedHeight"] = if (rotation == 90 || rotation == 270) width else height

        // Task 4: YOLO
        returnMap["yoloTimeMs"] = yoloTimeMs
        returnMap["yoloConfidence"] = maxConf.toDouble()
        returnMap["yoloClassId"] = 0
        returnMap["yoloBox"] = listOf(
            x1.toDouble() / yoloWidth.toDouble(),
            y1.toDouble() / yoloHeight.toDouble(),
            x2.toDouble() / yoloWidth.toDouble(),
            y2.toDouble() / yoloHeight.toDouble()
        )

        // Task 5: Crop ROI
        returnMap["cropTimeMs"] = cropResizeTimeMs
        returnMap["cropWidth"] = cropWidth
        returnMap["cropHeight"] = cropHeight

        // Task 6: Resize ROI
        returnMap["resizedWidth"] = 224
        returnMap["resizedHeight"] = 224
        returnMap["resizedInterpolation"] = "bilinear"

        // Crop JPEG bytes (Removed in optimization, UI will not show ROI preview)
        returnMap["roiBytes"] = ByteArray(0)

        // Task 7 & 8: MediaPipe & Validation
        returnMap["mpTimeMs"] = mpTimeMs
        returnMap["mpLandmarkCount"] = if (handDetected) 21 else 0
        returnMap["rawLandmarks"] = landmarksArray.toList()
        returnMap["landmarkMin"] = landmarkMin.toDouble()
        returnMap["landmarkMax"] = landmarkMax.toDouble()
        returnMap["landmarkAvg"] = landmarkAvg.toDouble()
        returnMap["hasNaN"] = hasNaN
        returnMap["hasInfinity"] = hasInfinite

        // Task 9: Landmark Buffer
        returnMap["sequenceLength"] = windowSize
        returnMap["landmarkBufferState"] = status
        returnMap["windowMotion"] = 0.0

        // Task 11: Tensor Generation
        returnMap["tensorShape"] = listOf(1, SEQUENCE_LENGTH, featureLength)
        returnMap["tensorDataType"] = "float32"
        returnMap["tensorFirstFrame"] = currentTensorFirstFrame
        returnMap["tensorLastFrame"] = currentTensorLastFrame

        // Task 12: LSTM
        returnMap["lstmTimeMs"] = currentLstmTimeMs
        returnMap["lstmOutputSize"] = 33
        returnMap["probabilities"] = currentMeanProbs.toList()

        // Task 14: Latency & final predicted indices
        returnMap["labelIndex"] = labelIndex
        returnMap["confidence"] = confidenceValue
        returnMap["totalTimeMs"] = totalTimeMs
        
        // Phase 1 Profiling Additions
        returnMap["threadCreationTimeMs"] = threadCreationTimeMs
        returnMap["yoloSkipped"] = yoloSkippedThisFrame
        returnMap["roiSmoothed"] = false

        // Phase 2 Profiling Additions
        returnMap["acquisitionDelayMs"] = currentAcquisitionDelayMs
        returnMap["yoloCoverage"] = currentYoloCoverage
        returnMap["motionToYoloDelayMs"] = currentMotionYoloDelayMs
        returnMap["motionToRoiDelayMs"] = currentMotionRoiDelayMs
        returnMap["motionToMpDelayMs"] = currentMotionMpDelayMs
        returnMap["motionToBufferDelayMs"] = currentMotionBufferDelayMs

        return returnMap
    }

    private fun printDropSummary(totalCameraFrames: Int, snapshotSize: Int, usedIndices: List<Int>, windowsGenerated: Int, validWindows: Int, gestureDuration: Long, finalLabel: Int, finalConf: Double) {
        sessionTotalCameraFrames = totalCameraFrames - startCameraFramesCount
        val totalCamera = if (sessionTotalCameraFrames > 0) sessionTotalCameraFrames else 0
        val totalDropped = dropDroppedBeforeYolo + dropRejectedYolo + dropFailedMediaPipe + dropRejectedSegmentation + dropDiscardedMotionFilter
        val dropRate = if (totalCamera > 0) (totalDropped.toDouble() / totalCamera) * 100.0 else 0.0

        android.util.Log.d("PIPELINE_VALIDATION", "===========================================")
        android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId] GESTURE-LEVEL PROFILING SUMMARY")
        android.util.Log.d("PIPELINE_VALIDATION", "Gesture ID               : $currentGestureId")
        android.util.Log.d("PIPELINE_VALIDATION", "Gesture Duration         : $gestureDuration ms")
        android.util.Log.d("PIPELINE_VALIDATION", "Total Camera Frames      : $totalCamera")
        android.util.Log.d("PIPELINE_VALIDATION", "Frames Entering Kotlin   : $totalCamera")
        android.util.Log.d("PIPELINE_VALIDATION", "Frames Processed by YOLO : $sessionProcessedByYolo")
        android.util.Log.d("PIPELINE_VALIDATION", "Frames with Landmarks    : $sessionWithLandmarks")
        android.util.Log.d("PIPELINE_VALIDATION", "Frames Added to Buffer   : $snapshotSize")
        android.util.Log.d("PIPELINE_VALIDATION", "Frozen Buffer Size       : $snapshotSize")
        android.util.Log.d("PIPELINE_VALIDATION", "Frames Used by LSTM      : ${usedIndices.size}")
        android.util.Log.d("PIPELINE_VALIDATION", "Sliding Windows Generated: $windowsGenerated")
        android.util.Log.d("PIPELINE_VALIDATION", "Valid Windows            : $validWindows")
        android.util.Log.d("PIPELINE_VALIDATION", "Rejected Windows         : ${windowsGenerated - validWindows}")
        android.util.Log.d("PIPELINE_VALIDATION", "Final Label              : $finalLabel")
        android.util.Log.d("PIPELINE_VALIDATION", "Final Confidence         : ${String.format(Locale.US, "%.4f", finalConf)}")
        android.util.Log.d("PIPELINE_VALIDATION", "--- Backpressure Statistics ---")
        android.util.Log.d("PIPELINE_VALIDATION", "Total Fully Processed    : $sessionWithLandmarks")
        android.util.Log.d("PIPELINE_VALIDATION", "Total Dropped Frames     : $totalDropped")
        android.util.Log.d("PIPELINE_VALIDATION", "Drop Rate                : ${String.format(Locale.US, "%.2f", dropRate)}%")
        val avgQWait = if (latencyCount > 0) ((accumCameraToYolo / latencyCount) + (accumYoloToMp / latencyCount) + (accumMpToBuffer / latencyCount)) else 0.0
        android.util.Log.d("PIPELINE_VALIDATION", "Average Queue Waiting    : ${String.format(Locale.US, "%.2f", avgQWait)} ms")
        val effectiveFps = if (gestureDuration > 0) (sessionWithLandmarks * 1000.0 / gestureDuration) else 0.0
        android.util.Log.d("PIPELINE_VALIDATION", "Effective Processing FPS : ${String.format(Locale.US, "%.2f", effectiveFps)}")
        android.util.Log.d("PIPELINE_VALIDATION", "")
        android.util.Log.d("PIPELINE_VALIDATION", "Drops Breakdown:")
        android.util.Log.d("PIPELINE_VALIDATION", "- Dropped before YOLO (Locks/Throttling): $dropDroppedBeforeYolo")
        android.util.Log.d("PIPELINE_VALIDATION", "- Rejected by YOLO (Low confidence): $dropRejectedYolo")
        android.util.Log.d("PIPELINE_VALIDATION", "- Failed MediaPipe (No landmarks): $dropFailedMediaPipe")
        android.util.Log.d("PIPELINE_VALIDATION", "- Rejected by Segmentation (Timeout/Noise): $dropRejectedSegmentation")
        android.util.Log.d("PIPELINE_VALIDATION", "- Discarded by Motion Filtering: $dropDiscardedMotionFilter")
        android.util.Log.d("PIPELINE_VALIDATION", "--- Session Summary Metrics ---")
        android.util.Log.d("PIPELINE_VALIDATION", "Completed Sessions: $totalCompletedSessions")
        android.util.Log.d("PIPELINE_VALIDATION", "Discarded Sessions: $totalDiscardedSessions")
        android.util.Log.d("PIPELINE_VALIDATION", "Cancelled Freezes : $cancelledFreezeAttempts")
        android.util.Log.d("PIPELINE_VALIDATION", "Inference Sessions: $completedInferenceSessions")
        android.util.Log.d("PIPELINE_VALIDATION", "High Confidence   : $highConfidencePredictions")
        android.util.Log.d("PIPELINE_VALIDATION", "===========================================")
    }

    private fun printLatencySummary(lstmTimeMs: Double) {
        val count = if (latencyCount > 0) latencyCount else 1
        val avgTotalPrep = accumTotalPrep / count
        val totalInferenceMs = lstmTimeMs
        val endToEndMs = avgTotalPrep + totalInferenceMs

        android.util.Log.d("PIPELINE_VALIDATION", "===========================================")
        android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId] LATENCY DISTRIBUTION & PERCENTILE ANALYSIS")
        printStageStats("Camera Image Conversion", latConversion)
        printStageStats("Tensor Preparation", latTensorPrep)
        printStageStats("YOLO Inference", latYoloInference)
        printStageStats("YOLO Postprocessing", latYoloPost)
        printStageStats("MediaPipe", latMediaPipe)
        printStageStats("Total Preprocessing", latTotalPrep)
        
        android.util.Log.d("PIPELINE_VALIDATION", "---")
        android.util.Log.d("PIPELINE_VALIDATION", "Average End-to-End Latency: ${String.format(Locale.US, "%.2f", endToEndMs)} ms")
        android.util.Log.d("PIPELINE_VALIDATION", "Average Total Preprocessing: ${String.format(Locale.US, "%.2f", avgTotalPrep)} ms")
        android.util.Log.d("PIPELINE_VALIDATION", "Total LSTM Inference Time: ${String.format(Locale.US, "%.2f", lstmTimeMs)} ms")
        android.util.Log.d("PIPELINE_VALIDATION", "===========================================")
    }

    private fun calculateIoU(box1: FloatArray, box2: FloatArray): Float {
        val x1_1 = box1[0] - box1[2] / 2f
        val y1_1 = box1[1] - box1[3] / 2f
        val x2_1 = box1[0] + box1[2] / 2f
        val y2_1 = box1[1] + box1[3] / 2f

        val x1_2 = box2[0] - box2[2] / 2f
        val y1_2 = box2[1] - box2[3] / 2f
        val x2_2 = box2[0] + box2[2] / 2f
        val y2_2 = box2[1] + box2[3] / 2f

        val xi1 = maxOf(x1_1, x1_2)
        val yi1 = maxOf(y1_1, y1_2)
        val xi2 = minOf(x2_1, x2_2)
        val yi2 = minOf(y2_1, y2_2)

        val interArea = maxOf(0.0f, xi2 - xi1) * maxOf(0.0f, yi2 - yi1)
        val box1Area = box1[2] * box1[3]
        val box2Area = box2[2] * box2[3]
        val unionArea = box1Area + box2Area - interArea

        return if (unionArea > 0f) interArea / unionArea else 0.0f
    }

    private fun yuv420ToNv21(width: Int, height: Int, planes: List<Map<String, Any>>): ByteArray {
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBytes = yPlane["bytes"] as ByteArray
        val uBytes = uPlane["bytes"] as ByteArray
        val vBytes = vPlane["bytes"] as ByteArray

        val yBytesPerRow = (yPlane["bytesPerRow"] as Number).toInt()
        val uBytesPerRow = (uPlane["bytesPerRow"] as Number).toInt()
        val vBytesPerRow = (vPlane["bytesPerRow"] as Number).toInt()

        val uPixelStride = (uPlane["bytesPerPixel"] as Number?)?.toInt() ?: 1
        val vPixelStride = (vPlane["bytesPerPixel"] as Number?)?.toInt() ?: 1

        val nv21 = ByteArray(width * height * 3 / 2)

        var idY = 0
        for (row in 0 until height) {
            val rowOffset = row * yBytesPerRow
            System.arraycopy(yBytes, rowOffset, nv21, idY, width)
            idY += width
        }

        var idUV = width * height
        val uvWidth = width / 2
        val uvHeight = height / 2

        for (row in 0 until uvHeight) {
            val uRowOffset = row * uBytesPerRow
            val vRowOffset = row * vBytesPerRow
            for (col in 0 until uvWidth) {
                val uIdx = uRowOffset + col * uPixelStride
                val vIdx = vRowOffset + col * vPixelStride
                nv21[idUV++] = vBytes[vIdx]
                nv21[idUV++] = uBytes[uIdx]
            }
        }

        return nv21
    }

    /**
     * Computes the average per-landmark per-frame Euclidean motion for a 30-frame window.
     * Mathematically equivalent to Streamlit calculate_window_motion().
     * Only considers frames with non-zero landmark data (to skip zero-padded frames).
     */
    private fun calculateWindowMotion(window: Array<FloatArray>): Float {
        val validFrames = ArrayList<FloatArray>()
        for (frame in window) {
            var sumAbs = 0.0f
            for (v in frame) sumAbs += Math.abs(v)
            if (sumAbs > 0f) validFrames.add(frame)
        }
        if (validFrames.size < 2) return 0.0f

        var totalDist = 0.0f
        val numDiffs = validFrames.size - 1
        for (i in 0 until numDiffs) {
            val curr = validFrames[i]
            val next = validFrames[i + 1]
            var frameDist = 0.0f
            for (j in 0 until 21) {
                val dx = next[j * 3]     - curr[j * 3]
                val dy = next[j * 3 + 1] - curr[j * 3 + 1]
                val dz = next[j * 3 + 2] - curr[j * 3 + 2]
                frameDist += Math.sqrt((dx * dx + dy * dy + dz * dz).toDouble()).toFloat()
            }
            totalDist += frameDist / 21f
        }
        return totalDist / numDiffs
    }

    private fun calculatePercentile(data: List<Double>, percentile: Double): Double {
        if (data.isEmpty()) return 0.0
        val sorted = data.sorted()
        val index = Math.ceil((percentile / 100.0) * sorted.size).toInt() - 1
        return sorted[maxOf(0, index)]
    }

    private fun printStageStats(stageName: String, data: List<Double>) {
        if (data.isEmpty()) return
        val sorted = data.sorted()
        val min = sorted.first()
        val max = sorted.last()
        val mean = data.sum() / data.size
        val median = if (sorted.size % 2 == 0) (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2.0 else sorted[sorted.size / 2]
        var variance = 0.0
        for (v in data) variance += (v - mean) * (v - mean)
        val stdDev = Math.sqrt(variance / data.size)
        val p95 = calculatePercentile(data, 95.0)
        val p99 = calculatePercentile(data, 99.0)

        android.util.Log.d("PIPELINE_VALIDATION", "[$sessionId][$currentGestureId] $stageName | Mean: ${String.format(Locale.US, "%.2f", mean)} | Median: ${String.format(Locale.US, "%.2f", median)} | Min: ${String.format(Locale.US, "%.2f", min)} | Max: ${String.format(Locale.US, "%.2f", max)} | StdDev: ${String.format(Locale.US, "%.2f", stdDev)} | P95: ${String.format(Locale.US, "%.2f", p95)} | P99: ${String.format(Locale.US, "%.2f", p99)}")
    }

    override fun onDestroy() {
        super.onDestroy()
        yoloInterpreter?.close()
        handLandmarker?.close()
        lstmInterpreter?.close()
    }

    private fun convertYuvToFloatBuffer(
        planes: List<Map<String, Any>>,
        srcWidth: Int,
        srcHeight: Int,
        rotation: Int,
        dstWidth: Int,
        dstHeight: Int,
        outputBuffer: ByteBuffer
    ) {
        val yPlane = planes[0]["bytes"] as ByteArray
        val uPlane = planes[1]["bytes"] as ByteArray
        val vPlane = planes[2]["bytes"] as ByteArray

        val yRowStride = planes[0]["bytesPerRow"] as Int
        val yPixelStride = planes[0]["bytesPerPixel"] as Int

        val uvRowStride = planes[1]["bytesPerRow"] as Int
        val uvPixelStride = planes[1]["bytesPerPixel"] as Int

        val rotatedWidth = if (rotation == 90 || rotation == 270) srcHeight else srcWidth
        val rotatedHeight = if (rotation == 90 || rotation == 270) srcWidth else srcHeight

        outputBuffer.rewind()

        for (dstY in 0 until dstHeight) {
            for (dstX in 0 until dstWidth) {
                val rx = (dstX * rotatedWidth) / dstWidth
                val ry = (dstY * rotatedHeight) / dstHeight

                val srcX: Int
                val srcY: Int
                when (rotation) {
                    90 -> {
                        srcX = ry
                        srcY = srcHeight - 1 - rx
                    }
                    180 -> {
                        srcX = srcWidth - 1 - rx
                        srcY = srcHeight - 1 - ry
                    }
                    270 -> {
                        srcX = srcWidth - 1 - ry
                        srcY = rx
                    }
                    else -> {
                        srcX = rx
                        srcY = ry
                    }
                }

                val cx = srcX.coerceIn(0, srcWidth - 1)
                val cy = srcY.coerceIn(0, srcHeight - 1)

                val yIdx = cy * yRowStride + cx * yPixelStride
                val uvIdx = (cy / 2) * uvRowStride + (cx / 2) * uvPixelStride

                val y = (yPlane[yIdx].toInt() and 0xFF)
                val u = (uPlane[uvIdx].toInt() and 0xFF) - 128
                val v = (vPlane[uvIdx].toInt() and 0xFF) - 128

                var r = y + (1.370705f * v)
                var g = y - (0.337633f * u) - (0.698001f * v)
                var b = y + (1.732446f * u)

                r = r.coerceIn(0f, 255f)
                g = g.coerceIn(0f, 255f)
                b = b.coerceIn(0f, 255f)

                outputBuffer.putFloat(r / 255.0f)
                outputBuffer.putFloat(g / 255.0f)
                outputBuffer.putFloat(b / 255.0f)
            }
        }
    }

    private fun extractRoiToBitmap(
        planes: List<Map<String, Any>>,
        srcWidth: Int,
        srcHeight: Int,
        rotation: Int,
        roiX: Int,
        roiY: Int,
        roiW: Int,
        roiH: Int,
        dstBitmap: Bitmap,
        dstIntArray: IntArray
    ) {
        val yPlane = planes[0]["bytes"] as ByteArray
        val uPlane = planes[1]["bytes"] as ByteArray
        val vPlane = planes[2]["bytes"] as ByteArray

        val yRowStride = planes[0]["bytesPerRow"] as Int
        val yPixelStride = planes[0]["bytesPerPixel"] as Int

        val uvRowStride = planes[1]["bytesPerRow"] as Int
        val uvPixelStride = planes[1]["bytesPerPixel"] as Int

        val rotatedWidth = if (rotation == 90 || rotation == 270) srcHeight else srcWidth
        val rotatedHeight = if (rotation == 90 || rotation == 270) srcWidth else srcHeight
        
        val dstWidth = dstBitmap.width
        val dstHeight = dstBitmap.height

        var idx = 0
        for (dstY in 0 until dstHeight) {
            for (dstX in 0 until dstWidth) {
                // Map dstX, dstY to bounding box in 640x640 space
                val bx = roiX + (dstX * roiW) / dstWidth
                val by = roiY + (dstY * roiH) / dstHeight
                
                // Map bx, by to rotated original space
                val rx = (bx * rotatedWidth) / 640
                val ry = (by * rotatedHeight) / 640

                val srcX: Int
                val srcY: Int
                when (rotation) {
                    90 -> {
                        srcX = ry
                        srcY = srcHeight - 1 - rx
                    }
                    180 -> {
                        srcX = srcWidth - 1 - rx
                        srcY = srcHeight - 1 - ry
                    }
                    270 -> {
                        srcX = srcWidth - 1 - ry
                        srcY = rx
                    }
                    else -> {
                        srcX = rx
                        srcY = ry
                    }
                }

                val cx = srcX.coerceIn(0, srcWidth - 1)
                val cy = srcY.coerceIn(0, srcHeight - 1)

                val yIdx = cy * yRowStride + cx * yPixelStride
                val uvIdx = (cy / 2) * uvRowStride + (cx / 2) * uvPixelStride

                val y = (yPlane[yIdx].toInt() and 0xFF)
                val u = (uPlane[uvIdx].toInt() and 0xFF) - 128
                val v = (vPlane[uvIdx].toInt() and 0xFF) - 128

                var r = (y + (1.370705f * v)).toInt()
                var g = (y - (0.337633f * u) - (0.698001f * v)).toInt()
                var b = (y + (1.732446f * u)).toInt()

                r = r.coerceIn(0, 255)
                g = g.coerceIn(0, 255)
                b = b.coerceIn(0, 255)

                dstIntArray[idx++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        dstBitmap.setPixels(dstIntArray, 0, dstWidth, 0, 0, dstWidth, dstHeight)
    }
}
