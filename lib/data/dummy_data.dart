import '../models/history_model.dart';

class DummyData {
  static List<String> translations = [
    'Halo',
    'Selamat Pagi',
    'Apa Kabar',
  ];

  static const dummyEmail = 'admin@1.com';
  static const dummyPassword = '1234';

  static List<HistoryModel> history = [
    HistoryModel(
      text: 'Halo',
      timestamp: DateTime.now().subtract(const Duration(minutes: 10)),
    ),
  ];
}
