class PredictionResult {
  const PredictionResult({
    required this.label,
    required this.confidence,
    this.handBox,
  });

  final String label;
  final double confidence;
  final List<double>? handBox;
}
