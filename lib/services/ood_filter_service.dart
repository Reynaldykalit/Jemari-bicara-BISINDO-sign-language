import 'dart:math';

/// Result container for the OOD (Out-of-Distribution) & Rejection Filter
class OodFilterResult {
  final bool isAccepted;
  final String statusMessage;
  final String rejectReason;
  final int? overrideLabelIndex;

  const OodFilterResult({
    required this.isAccepted,
    required this.statusMessage,
    required this.rejectReason,
    this.overrideLabelIndex,
  });
}

class OodFilterService {
  // Global Base Confidence Threshold (35% ramah untuk video real-time)
  static const double baseConfidenceThreshold = 0.35;

  // Confusion Pairs: Pasangan kata yang secara fisik/visual sangat mirip (Berdasarkan idx2label di label_map.json)
  static const List<Set<int>> confusionPairIndices = [
    {27, 1}, // Terima kasih (27) vs Mengapa (1)
    {20, 28}, // Ingat (20) vs Tuli (28) - Gerakan 1 jari di pelipis/telinga
    {8, 28}, // Dengar (8) vs Tuli (28) - Gerakan 1 jari menunjuk telinga
    {8, 20}, // Dengar (8) vs Ingat (20) - Gerakan 1 jari di area kepala/telinga
    {18, 19}, // Malam (18) vs Hari (19) - Gerakan menyilang di wajah/dada
    {14, 18}, // Pagi (14) vs Malam (18) - Gerakan lintasan telapak tangan
    {14, 15}, // Pagi (14) vs Siang (15)
    {15, 17}, // Siang (15) vs Sore (17)
    {14, 17}, // Pagi (14) vs Sore (17)
    {29, 30}, // Apa (29) vs Siapa (30)
  ];

  /// Utama: Evaluasi apakah prediksi dari LSTM & MediaPipe layak diterima atau ditolak
  static OodFilterResult evaluatePrediction({
    required int labelIndex,
    required double confidence,
    required List<double> rawLandmarks,
    List<double>? probabilities,
    List<double>? tensorLastFrame,
  }) {
    // 1. Jika labelIndex bernilai -1 (orang lewat / gerakan tidak sengaja)
    if (labelIndex == -1) {
      return const OodFilterResult(
        isAccepted: false,
        statusMessage: 'PASSBY_MOTION',
        rejectReason:
            'Native pipeline motion filter rejected (Ambient / Pass-by motion)',
      );
    }

    // Tentukan landmark mana yang valid (Fallback ke tensorLastFrame jika rawLandmarks kosong/all zero)
    List<double> effectiveLandmarks = rawLandmarks;
    if (_isLandmarkArrayEmpty(effectiveLandmarks) &&
        tensorLastFrame != null &&
        !_isLandmarkArrayEmpty(tensorLastFrame)) {
      effectiveLandmarks = tensorLastFrame;
    }

    // 2. Jika landmark kosong atau seluruh nilainya 0 (orang lewat / tidak ada tangan di kamera)
    if (_isLandmarkArrayEmpty(effectiveLandmarks)) {
      return const OodFilterResult(
        isAccepted: false,
        statusMessage: 'PASSBY_MOTION',
        rejectReason:
            'No valid hand landmarks detected by MediaPipe (Pass-by / Accidental)',
      );
    }

    // 3. Postural Remap Akurat untuk Pasangan Mirip: Ingat (20) vs Tuli (28), dan Makan (23)
    if (effectiveLandmarks.length == 63) {
      double dist3d(int i, int j) {
        double dx = effectiveLandmarks[i * 3] - effectiveLandmarks[j * 3];
        double dy =
            effectiveLandmarks[i * 3 + 1] - effectiveLandmarks[j * 3 + 1];
        double dz =
            effectiveLandmarks[i * 3 + 2] - effectiveLandmarks[j * 3 + 2];
        return sqrt(dx * dx + dy * dy + dz * dz);
      }

      final double handScale = dist3d(0, 9);
      if (handScale >= 0.01) {
        final double indexExt = dist3d(0, 8) / handScale;

        // Jika diprediksi Ingat (20) tapi Telunjuk TERENTANG LURUS (> 1.40) -> Ini adalah TULI (28)!
        if (labelIndex == 20 && indexExt > 1.40) {
          return const OodFilterResult(
            isAccepted: true,
            statusMessage: 'Hasil Ditemukan',
            rejectReason:
                'Postural Remap: Ingat -> Tuli (Index finger extended straight out)',
            overrideLabelIndex: 28,
          );
        }

        // Jika diprediksi Tuli (28) tapi Telunjuk TERTEKUK (< 1.30) -> Ini adalah INGAT (20)!
        if (labelIndex == 28 && indexExt < 1.30) {
          return const OodFilterResult(
            isAccepted: true,
            statusMessage: 'Hasil Ditemukan',
            rejectReason:
                'Postural Remap: Tuli -> Ingat (Knuckle fist at temple)',
            overrideLabelIndex: 20,
          );
        }

        // Khusus Gerakan MAKAN (23): Jari Kuncup di Depan Mulut/Wajah
        final double middleExt = dist3d(0, 12) / handScale;

        final bool isPinched = indexExt < 1.55 && middleExt < 1.55;

        if (isPinched && (labelIndex == 1 || labelIndex == 7 || labelIndex == 27)) {
          return const OodFilterResult(
            isAccepted: true,
            statusMessage: 'Hasil Ditemukan',
            rejectReason:
                'Postural Remap: -> Makan (Pinched fingers posture)',
            overrideLabelIndex: 23,
          );
        }

        // Hari (19) vs Hitam (7) Postural Remap (Sesuai Foto Dataset):
        // Hitam (7): Dari telapak terbuka di dahi ditarik turun mengepal (ends with closed fist)
        // Hari (19): Telapak tangan tetap terbuka (stays open palm)
        final double ringExt = dist3d(0, 16) / handScale;

        final bool isFistOrFolded =
            indexExt < 1.35 && middleExt < 1.35 && ringExt < 1.35;
        final bool isOpenPalm =
            indexExt > 1.40 && middleExt > 1.40 && ringExt > 1.40;

        // Jika diprediksi Hari (19), Kuning (4), atau Merah (3) tapi tangan MENGEPAL (fist/folded) -> Ini adalah HITAM (7)!
        if ((labelIndex == 19 || labelIndex == 4 || labelIndex == 3) &&
            isFistOrFolded) {
          return const OodFilterResult(
            isAccepted: true,
            statusMessage: 'Hasil Ditemukan',
            rejectReason:
                'Postural Remap: -> Hitam (Ends with closed fist posture)',
            overrideLabelIndex: 7,
          );
        }

        // Jika diprediksi Hitam (7) tapi tangan TETAP TELAPAK TERBUKA -> Ini adalah HARI (19)!
        if (labelIndex == 7 && isOpenPalm) {
          return const OodFilterResult(
            isAccepted: true,
            statusMessage: 'Hasil Ditemukan',
            rejectReason:
                'Postural Remap: Hitam -> Hari (Stays open palm posture)',
            overrideLabelIndex: 19,
          );
        }
      }
    }

    // 4. Evaluasi Biomechanical Posture (Hanya menolak OOD eksplisit non-BISINDO)
    if (effectiveLandmarks.length == 63) {
      final postureResult = _evaluateHandPosture(
        effectiveLandmarks,
        labelIndex,
      );
      if (postureResult != null) {
        return postureResult;
      }
    }

    // 5. Margin Gap & Pasangan Kata Mirip (Hari vs Malam, Pagi vs Malam, Apa vs Siapa, dll)
    double top1Prob = confidence;
    int top2Idx = -1;

    if (probabilities != null && probabilities.length >= 33) {
      final sortedIndices = List<int>.generate(probabilities.length, (i) => i)
        ..sort((a, b) => probabilities[b].compareTo(probabilities[a]));

      top1Prob = probabilities[sortedIndices[0]];
      top2Idx = sortedIndices[1];
    }

    bool isConfusedPair = false;
    if (top2Idx != -1) {
      for (final pair in confusionPairIndices) {
        if (pair.contains(labelIndex) && pair.contains(top2Idx)) {
          isConfusedPair = true;
          break;
        }
      }
    }

    // Untuk Pasangan Kata Mirip: Hanya menolak jika confidence sangat rendah (< 30%)
    if (isConfusedPair) {
      if (top1Prob < 0.30) {
        return OodFilterResult(
          isAccepted: false,
          statusMessage: 'PASSBY_MOTION',
          rejectReason:
              'Inter-Class Confusion Pair ($labelIndex vs $top2Idx, low confidence)',
        );
      }
    }

    // 6. SEMUA GERAKAN DATASET (Confidence >= 35%) -> LANGSUNG DITERIMA DENGAN MULUS!
    if (confidence >= baseConfidenceThreshold) {
      return OodFilterResult(
        isAccepted: true,
        statusMessage: 'Hasil Ditemukan',
        rejectReason:
            'Accepted by Confidence (Prob: ${(confidence * 100).toStringAsFixed(1)}%)',
      );
    }

    return const OodFilterResult(
      isAccepted: false,
      statusMessage: 'PASSBY_MOTION',
      rejectReason: 'Low confidence ambient motion',
    );
  }

  /// Memeriksa postur spesifik jari (Hanya menolak OOD eksplisit non-BISINDO)
  static OodFilterResult? _evaluateHandPosture(
    List<double> lm,
    int labelIndex,
  ) {
    if (lm.length < 63) return null;

    double dist3d(int i, int j) {
      double dx = lm[i * 3] - lm[j * 3];
      double dy = lm[i * 3 + 1] - lm[j * 3 + 1];
      double dz = lm[i * 3 + 2] - lm[j * 3 + 2];
      return sqrt(dx * dx + dy * dy + dz * dz);
    }

    final double handScale = dist3d(0, 9);
    if (handScale < 0.01) return null;

    final double indexExt = dist3d(0, 8) / handScale;
    final double middleExt = dist3d(0, 12) / handScale;
    final double ringExt = dist3d(0, 16) / handScale;
    final double pinkyExt = dist3d(0, 20) / handScale;

    // 1. Middle Finger / Fuck Sign (🖕)
    final bool isMiddleFingerSign =
        middleExt > 1.60 &&
        indexExt < 1.40 &&
        ringExt < 1.40 &&
        pinkyExt < 1.40;

    if (isMiddleFingerSign) {
      return const OodFilterResult(
        isAccepted: false,
        statusMessage: 'Bukan gerakan BISINDO',
        rejectReason: 'Out-of-Distribution: Middle Finger Sign (🖕) Detected',
      );
    }

    // 2. Metal / Rock Sign (🤘)
    final bool isMetalSign =
        indexExt > 1.55 &&
        pinkyExt > 1.45 &&
        middleExt < 1.40 &&
        ringExt < 1.40;

    if (isMetalSign) {
      return const OodFilterResult(
        isAccepted: false,
        statusMessage: 'Bukan gerakan BISINDO',
        rejectReason: 'Out-of-Distribution: Metal Sign (🤘) Detected',
      );
    }

    // 3. Peace / V Sign (✌️)
    final bool isPeaceSign =
        indexExt > 1.55 &&
        middleExt > 1.55 &&
        ringExt < 1.40 &&
        pinkyExt < 1.40;

    if (isPeaceSign && labelIndex != 32) {
      return const OodFilterResult(
        isAccepted: false,
        statusMessage: 'Bukan gerakan BISINDO',
        rejectReason: 'Out-of-Distribution: Peace Sign (✌️) Detected',
      );
    }

    return null; // Posture valid untuk seluruh gerakan BISINDO!
  }

  /// Pengecekan apakah array landmark kosong atau seluruh komponennya 0.0
  static bool _isLandmarkArrayEmpty(List<double> lm) {
    if (lm.isEmpty) return true;
    for (int i = 0; i < lm.length; i++) {
      if (lm[i] != 0.0) return false;
    }
    return true;
  }
}
