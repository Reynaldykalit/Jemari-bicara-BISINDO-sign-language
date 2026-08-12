import 'package:flutter/material.dart';
import 'package:jemari_bicara/screens/splash_screen.dart';
import 'package:jemari_bicara/theme/app_theme.dart';

void main() {
  runApp(const JemariBicaraApp());
}

class JemariBicaraApp extends StatelessWidget {
  const JemariBicaraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<ThemeMode>(
      valueListenable: AppTheme.themeNotifier,
      builder: (context, currentMode, _) {
        return MaterialApp(
          title: 'Jemari Bicara',
          debugShowCheckedModeBanner: false,
          theme: AppTheme.lightTheme,
          darkTheme: AppTheme.darkTheme,
          themeMode: currentMode,
          home: const SplashScreen(),
        );
      },
    );
  }
}
