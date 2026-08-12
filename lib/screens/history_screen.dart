import 'package:flutter/material.dart';
import '../data/dummy_data.dart';
import '../widgets/history_card.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  void clearAll() {
    setState(() {
      DummyData.history.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    final history = DummyData.history;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Riwayat Terjemahan'),
        actions: [
          IconButton(
            onPressed: clearAll,
            icon: const Icon(Icons.delete_forever),
          ),
        ],
      ),
      body: history.isEmpty
          ? const Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.history, size: 80, color: Colors.grey),
                  SizedBox(height: 16),
                  Text(
                    'Belum ada riwayat terjemahan',
                    style: TextStyle(fontSize: 16),
                  ),
                ],
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(20),
              itemCount: history.length,
              itemBuilder: (context, index) {
                return HistoryCard(item: history[index]);
              },
            ),
    );
  }
}
