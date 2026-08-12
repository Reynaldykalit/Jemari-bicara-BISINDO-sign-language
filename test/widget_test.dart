// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter_test/flutter_test.dart';

import 'package:jemari_bicara/main.dart';

void main() {
  testWidgets('Jemari Bicara app opens splash screen', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const JemariBicaraApp());

    expect(find.text('Jemari Bicara'), findsWidgets);
    expect(find.text('Ketuk layar untuk masuk'), findsOneWidget);
  });
}
