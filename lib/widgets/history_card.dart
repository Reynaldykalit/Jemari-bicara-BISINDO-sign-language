import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../models/history_model.dart';

class HistoryCard extends StatelessWidget {
  final HistoryModel item;

  const HistoryCard({super.key, required this.item});

  @override
  Widget build(BuildContext context) {
    final formatted = DateFormat('dd MMM yyyy, HH:mm').format(item.timestamp);

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            formatted,
            style: TextStyle(
              color: Theme.of(context).brightness == Brightness.dark
                  ? Colors.grey.shade400
                  : Colors.grey.shade600,
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            item.text,
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
          ),
        ],
      ),
    );
  }
}
