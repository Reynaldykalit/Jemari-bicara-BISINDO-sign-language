class LandmarkSequenceBuffer {
  LandmarkSequenceBuffer({this.sequenceLength = 10, this.featureLength = 63});

  final int sequenceLength;
  final int featureLength;
  final List<List<double>> _frames = [];

  bool get isReady => _frames.length >= sequenceLength;

  int get length => _frames.length;

  void clear() {
    _frames.clear();
  }

  void add(List<double> landmarks) {
    if (landmarks.length != featureLength) {
      throw ArgumentError(
        'Expected $featureLength landmark values, got ${landmarks.length}.',
      );
    }

    _frames.add(List<double>.from(landmarks));
    while (_frames.length > sequenceLength) {
      _frames.removeAt(0);
    }
  }

  List<List<double>> toSequence() {
    if (!isReady) {
      throw StateError('Sequence is not ready yet.');
    }

    return _frames
        .skip(_frames.length - sequenceLength)
        .map((frame) => List<double>.from(frame))
        .toList(growable: false);
  }
}
