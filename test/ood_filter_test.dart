import 'package:flutter_test/flutter_test.dart';
import 'package:jemari_bicara/services/ood_filter_service.dart';

void main() {
  group('OodFilterService Tests', () {
    List<double> createMockLandmarks({
      required double indexLength,
      required double middleLength,
      required double ringLength,
      required double pinkyLength,
      double wristY = 0.50,
      double middleMcpY = 0.30,
      double wristX = 0.50,
      double middleMcpX = 0.50,
    }) {
      final List<double> lm = List<double>.filled(63, 0.0);
      for (int i = 0; i < 21; i++) {
        lm[i * 3] = wristX;
        lm[i * 3 + 1] = (wristY + middleMcpY) / 2;
      }

      lm[0 * 3] = wristX;
      lm[0 * 3 + 1] = wristY;
      lm[0 * 3 + 2] = 0.0;

      lm[9 * 3] = middleMcpX;
      lm[9 * 3 + 1] = middleMcpY;
      lm[9 * 3 + 2] = 0.0;

      final double handScale = (middleMcpY - wristY).abs();
      final double effectiveScale = handScale < 0.05 ? 0.20 : handScale;

      lm[8 * 3] = wristX - 0.3 * effectiveScale;
      lm[8 * 3 + 1] = wristY - indexLength * effectiveScale;
      lm[8 * 3 + 2] = 0.0;

      lm[12 * 3] = wristX - 0.1 * effectiveScale;
      lm[12 * 3 + 1] = wristY - middleLength * effectiveScale;
      lm[12 * 3 + 2] = 0.0;

      lm[16 * 3] = wristX + 0.1 * effectiveScale;
      lm[16 * 3 + 1] = wristY - ringLength * effectiveScale;
      lm[16 * 3 + 2] = 0.0;

      lm[20 * 3] = wristX + 0.3 * effectiveScale;
      lm[20 * 3 + 1] = wristY - pinkyLength * effectiveScale;
      lm[20 * 3 + 2] = 0.0;

      return lm;
    }

    test('Rejects Middle Finger Sign (🖕) as OOD', () {
      final fuckSignLandmarks = createMockLandmarks(
        indexLength: 1.1,
        middleLength: 1.8,
        ringLength: 1.1,
        pinkyLength: 1.1,
      );

      final result = OodFilterService.evaluatePrediction(
        labelIndex: 10,
        confidence: 0.85,
        rawLandmarks: fuckSignLandmarks,
      );

      expect(result.isAccepted, isFalse);
      expect(result.statusMessage, equals('Bukan gerakan BISINDO'));
    });

    test('Rejects Metal Sign (🤘) as OOD', () {
      final metalLandmarks = createMockLandmarks(
        indexLength: 1.8,
        middleLength: 1.1,
        ringLength: 1.1,
        pinkyLength: 1.6,
      );

      final result = OodFilterService.evaluatePrediction(
        labelIndex: 5,
        confidence: 0.90,
        rawLandmarks: metalLandmarks,
      );

      expect(result.isAccepted, isFalse);
      expect(result.statusMessage, equals('Bukan gerakan BISINDO'));
    });

    test('Accepts valid gestures at base confidence threshold >= 35%', () {
      final validLandmarks = createMockLandmarks(
        indexLength: 1.5,
        middleLength: 1.5,
        ringLength: 1.5,
        pinkyLength: 1.5,
      );

      final result = OodFilterService.evaluatePrediction(
        labelIndex: 19, // Hari
        confidence: 0.40,
        rawLandmarks: validLandmarks,
      );

      expect(result.isAccepted, isTrue);
      expect(result.statusMessage, equals('Hasil Ditemukan'));
    });

    test('Rejects prediction when hand landmarks are empty/untracked by MediaPipe', () {
      final resultEmpty = OodFilterService.evaluatePrediction(
        labelIndex: 10,
        confidence: 0.51,
        rawLandmarks: List<double>.filled(63, 0.0),
      );

      expect(resultEmpty.isAccepted, isFalse);
      expect(resultEmpty.statusMessage, equals('PASSBY_MOTION'));
    });

    test('Accepts open palm gestures smoothly', () {
      final openPalmLandmarks = createMockLandmarks(
        indexLength: 1.8,
        middleLength: 1.8,
        ringLength: 1.8,
        pinkyLength: 1.8,
      );

      final resultSaya = OodFilterService.evaluatePrediction(
        labelIndex: 25, // Saya
        confidence: 0.45,
        rawLandmarks: openPalmLandmarks,
      );

      expect(resultSaya.isAccepted, isTrue);
    });

    test('Remaps Ingat (20) to Tuli (28) when index finger is extended straight out', () {
      final straightIndexLandmarks = createMockLandmarks(
        indexLength: 1.6,
        middleLength: 1.1,
        ringLength: 1.1,
        pinkyLength: 1.1,
      );

      final result = OodFilterService.evaluatePrediction(
        labelIndex: 20, // Ingat
        confidence: 0.55,
        rawLandmarks: straightIndexLandmarks,
      );

      expect(result.isAccepted, isTrue);
      expect(result.overrideLabelIndex, equals(28)); // Remapped to Tuli!
    });

    test('Remaps Tuli (28) to Ingat (20) when index finger is folded', () {
      final foldedLandmarks = createMockLandmarks(
        indexLength: 1.1,
        middleLength: 1.1,
        ringLength: 1.1,
        pinkyLength: 1.1,
      );

      final result = OodFilterService.evaluatePrediction(
        labelIndex: 28, // Tuli
        confidence: 0.55,
        rawLandmarks: foldedLandmarks,
      );

      expect(result.isAccepted, isTrue);
      expect(result.overrideLabelIndex, equals(20)); // Remapped to Ingat!
    });

    test('Remaps Mengapa (1) to Makan (23) when fingers are pinched at mouth level', () {
      final pinchedLandmarks = createMockLandmarks(
        indexLength: 1.1,
        middleLength: 1.1,
        ringLength: 1.1,
        pinkyLength: 1.1,
        wristY: 0.55,
        middleMcpY: 0.35,
      );

      final result = OodFilterService.evaluatePrediction(
        labelIndex: 1, // Mengapa
        confidence: 0.55,
        rawLandmarks: pinchedLandmarks,
      );

      expect(result.isAccepted, isTrue);
      expect(result.overrideLabelIndex, equals(23)); // Remapped to Makan!
    });

    test('Remaps Hitam (7) to Hari (19) when open palm posture is detected', () {
      final openPalmLandmarks = createMockLandmarks(
        indexLength: 1.6,
        middleLength: 1.6,
        ringLength: 1.6,
        pinkyLength: 1.6,
      );

      final result = OodFilterService.evaluatePrediction(
        labelIndex: 7, // Hitam
        confidence: 0.55,
        rawLandmarks: openPalmLandmarks,
      );

      expect(result.isAccepted, isTrue);
      expect(result.overrideLabelIndex, equals(19)); // Remapped to Hari!
    });

    test('Remaps Hari (19) to Hitam (7) when closed fist posture is detected', () {
      final closedFistLandmarks = createMockLandmarks(
        indexLength: 1.1,
        middleLength: 1.1,
        ringLength: 1.1,
        pinkyLength: 1.1,
      );

      final result = OodFilterService.evaluatePrediction(
        labelIndex: 19, // Hari
        confidence: 0.55,
        rawLandmarks: closedFistLandmarks,
      );

      expect(result.isAccepted, isTrue);
      expect(result.overrideLabelIndex, equals(7)); // Remapped to Hitam!
    });

    test('Remaps Kuning (4) to Hitam (7) when closed fist posture is detected', () {
      final closedFistLandmarks = createMockLandmarks(
        indexLength: 1.1,
        middleLength: 1.1,
        ringLength: 1.1,
        pinkyLength: 1.1,
      );

      final result = OodFilterService.evaluatePrediction(
        labelIndex: 4, // Kuning
        confidence: 0.55,
        rawLandmarks: closedFistLandmarks,
      );

      expect(result.isAccepted, isTrue);
      expect(result.overrideLabelIndex, equals(7)); // Remapped to Hitam!
    });
  });
}
