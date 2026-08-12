import 'package:flutter/material.dart';
import 'dart:math' as math;
import 'history_screen.dart';
import 'realtime_screen.dart';
import 'splash_screen.dart';
import '../theme/app_theme.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  static const Color mainBlue = Color(0xFF6FADE6);
  static const Color deepBlue = Color(0xFF4E93CF);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Scaffold(
      body: IndexedStack(
        index: _selectedIndex,
        children: const [_HomeTab(), HistoryScreen(), _SettingsTab()],
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          boxShadow: [
            BoxShadow(
              color: isDark
                  ? Colors.black.withValues(alpha: 0.24)
                  : Colors.black.withValues(alpha: 0.06),
              blurRadius: 10,
              offset: const Offset(0, -2),
            ),
          ],
        ),
        child: BottomNavigationBar(
          currentIndex: _selectedIndex,
          onTap: (index) {
            setState(() {
              _selectedIndex = index;
            });
          },
          backgroundColor: isDark ? AppTheme.darkCard : Colors.white,
          selectedItemColor: isDark
              ? const Color(0xFF60A5FA)
              : const Color(0xFF236CFF),
          unselectedItemColor: const Color(0xFF94A3B8),
          selectedLabelStyle: const TextStyle(
            fontWeight: FontWeight.w700,
            fontSize: 13,
          ),
          unselectedLabelStyle: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 13,
          ),
          type: BottomNavigationBarType.fixed,
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.home_rounded),
              label: 'Beranda',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.menu_book_rounded),
              label: 'Riwayat',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.settings_rounded),
              label: 'Pengaturan',
            ),
          ],
        ),
      ),
    );
  }
}

class _HomeTab extends StatelessWidget {
  const _HomeTab();

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: SafeArea(
        child: SingleChildScrollView(
          physics: const BouncingScrollPhysics(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const _Header(),
              const SizedBox(height: 48),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    // Title
                    Text(
                      'Siap Menerjemahkan?',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: isDark ? Colors.white : const Color(0xFF1E293B),
                        fontSize: 26,
                        fontWeight: FontWeight.w800,
                        letterSpacing: -0.5,
                      ),
                    ),
                    const SizedBox(height: 12),

                    // Subtitle
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      child: Text(
                        'Arahkan kamera ke gerakan tangan untuk mulai menerjemahkan bahasa isyarat ke teks',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: isDark
                              ? const Color(0xFF94A3B8)
                              : const Color(0xFF64748B),
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                          height: 1.5,
                        ),
                      ),
                    ),
                    const SizedBox(height: 48),

                    // Wavy Camera Button (No animation)
                    _WavyPulseButton(
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => const RealtimeScreen(),
                          ),
                        );
                      },
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header();

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      height: 82,
      margin: EdgeInsets.zero,
      decoration: BoxDecoration(
        color: isDark ? AppTheme.darkCard : HomeScreen.mainBlue,
        borderRadius: const BorderRadius.only(
          bottomLeft: Radius.circular(24),
          bottomRight: Radius.circular(24),
        ),
        boxShadow: [
          BoxShadow(
            color: isDark
                ? Colors.black.withValues(alpha: 0.36)
                : Colors.black.withValues(alpha: 0.24),
            blurRadius: 10,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: const BorderRadius.only(
          bottomLeft: Radius.circular(24),
          bottomRight: Radius.circular(24),
        ),
        child: Stack(
          children: [
            Positioned.fill(
              child: DecoratedBox(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.centerLeft,
                    end: Alignment.centerRight,
                    colors: isDark
                        ? [AppTheme.darkCard, AppTheme.darkBackground]
                        : [
                            HomeScreen.mainBlue,
                            HomeScreen.deepBlue.withValues(alpha: 0.96),
                          ],
                  ),
                ),
              ),
            ),
            Positioned.fill(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(12, 0, 14, 0),
                child: Row(
                  children: [
                    Expanded(
                      flex: 4,
                      child: FittedBox(
                        fit: BoxFit.scaleDown,
                        alignment: Alignment.centerLeft,
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: const [
                            Text(
                              'Jemari',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 21,
                                fontWeight: FontWeight.w900,
                              ),
                            ),
                            SizedBox(width: 5),
                            Text(
                              'Bicara',
                              style: TextStyle(
                                color: Color(0xFFFFD85E),
                                fontSize: 21,
                                fontWeight: FontWeight.w900,
                              ),
                            ),
                            SizedBox(width: 4),
                            Text(
                              '✦',
                              style: TextStyle(
                                color: Color(0xFFFFD85E),
                                fontSize: 14,
                                fontWeight: FontWeight.w900,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const Spacer(),
                    Flexible(
                      flex: 3,
                      child: Container(
                        constraints: const BoxConstraints(maxWidth: 112),
                        height: 82,
                        child: Align(
                          alignment: Alignment.bottomCenter,
                          child: Transform.translate(
                            offset: const Offset(0, 1),
                            child: Image.asset(
                              'assets/finger1.png',
                              height: 76,
                              fit: BoxFit.contain,
                              alignment: Alignment.bottomCenter,
                              filterQuality: FilterQuality.high,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _WavyPulseButton extends StatelessWidget {
  final VoidCallback onTap;

  const _WavyPulseButton({required this.onTap});

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final primaryColor = isDark
        ? const Color(0xFF60A5FA)
        : const Color(0xFF236CFF);

    return GestureDetector(
      onTap: onTap,
      child: SizedBox(
        width: 220,
        height: 220,
        child: Stack(
          alignment: Alignment.center,
          children: [
            // Static Waves
            CustomPaint(
              size: const Size(220, 220),
              painter: _StaticWavyRipplePainter(
                color: primaryColor,
                baseRadius: 55,
                amplitude: 8,
                waveCount: 12,
              ),
            ),
            // Center button
            Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [primaryColor, primaryColor.withValues(alpha: 0.85)],
                ),
                boxShadow: [
                  BoxShadow(
                    color: primaryColor.withValues(alpha: 0.35),
                    blurRadius: 18,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: const Icon(
                Icons.camera_alt_outlined,
                color: Colors.white,
                size: 42,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StaticWavyRipplePainter extends CustomPainter {
  final Color color;
  final double baseRadius;
  final double amplitude;
  final int waveCount;

  _StaticWavyRipplePainter({
    required this.color,
    required this.baseRadius,
    required this.amplitude,
    required this.waveCount,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);

    // Draw 3 static concentric waves with decreasing opacities
    final waves = [
      _WaveConfig(radiusOffset: 45.0, opacity: 0.1),
      _WaveConfig(radiusOffset: 25.0, opacity: 0.25),
      _WaveConfig(radiusOffset: 8.0, opacity: 0.45),
    ];

    for (final wave in waves) {
      final currentRadius = baseRadius + wave.radiusOffset;
      // Make amplitude slightly larger for outer waves to look natural
      final currentAmplitude = amplitude * (1.0 + wave.radiusOffset / 60.0);

      final paint = Paint()
        ..color = color.withValues(alpha: wave.opacity)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      final fillPaint = Paint()
        ..color = color.withValues(alpha: wave.opacity * 0.2)
        ..style = PaintingStyle.fill;

      final path = Path();
      const pointsCount = 180;
      for (int i = 0; i <= pointsCount; i++) {
        final theta = (i * 2 * math.pi) / pointsCount;
        final r =
            currentRadius + currentAmplitude * math.cos(theta * waveCount);
        final x = center.dx + r * math.cos(theta);
        final y = center.dy + r * math.sin(theta);
        if (i == 0) {
          path.moveTo(x, y);
        } else {
          path.lineTo(x, y);
        }
      }
      path.close();

      canvas.drawPath(path, fillPaint);
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _StaticWavyRipplePainter oldDelegate) {
    return oldDelegate.color != color;
  }
}

class _WaveConfig {
  final double radiusOffset;
  final double opacity;

  _WaveConfig({required this.radiusOffset, required this.opacity});
}

class _SettingsTab extends StatefulWidget {
  const _SettingsTab();

  @override
  State<_SettingsTab> createState() => _SettingsTabState();
}

class _SettingsTabState extends State<_SettingsTab> {
  bool _isDarkModeEnabled = false;

  @override
  void initState() {
    super.initState();
    _isDarkModeEnabled = AppTheme.themeNotifier.value == ThemeMode.dark;
  }

  void showAboutApp(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    showDialog(
      context: context,
      builder: (_) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
          ),
          title: Text(
            'Tentang Aplikasi',
            style: TextStyle(
              fontWeight: FontWeight.w800,
              color: isDark ? Colors.white : const Color(0xFF1E293B),
            ),
          ),
          content: Text(
            'Jemari Bicara adalah aplikasi penerjemah gerakan BISINDO ke teks secara real-time untuk menjembatani komunikasi teman tuli dengan masyarakat luas.',
            style: TextStyle(
              fontSize: 14,
              color: isDark ? const Color(0xFF94A3B8) : const Color(0xFF64748B),
              height: 1.5,
              fontWeight: FontWeight.w500,
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              style: TextButton.styleFrom(
                foregroundColor: const Color(0xFF236CFF),
              ),
              child: const Text('Tutup'),
            ),
          ],
        );
      },
    );
  }

  void _handleLogout(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    showDialog(
      context: context,
      builder: (_) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
          ),
          title: Text(
            'Konfirmasi Keluar',
            style: TextStyle(
              fontWeight: FontWeight.w800,
              color: isDark ? Colors.white : const Color(0xFF1E293B),
            ),
          ),
          content: Text(
            'Apakah Anda yakin ingin keluar dari aplikasi?',
            style: TextStyle(
              fontSize: 14,
              color: isDark ? const Color(0xFF94A3B8) : const Color(0xFF64748B),
              fontWeight: FontWeight.w500,
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              style: TextButton.styleFrom(
                foregroundColor: const Color(0xFF64748B),
              ),
              child: const Text('Batal'),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(context); // Close dialog
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (_) => const SplashScreen()),
                );
              },
              style: TextButton.styleFrom(foregroundColor: Colors.redAccent),
              child: const Text('Keluar'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Pengaturan',
          style: TextStyle(
            fontWeight: FontWeight.w800,
            color: isDark ? Colors.white : const Color(0xFF1E293B),
          ),
        ),
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        elevation: 0,
        scrolledUnderElevation: 0,
      ),
      body: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
        child: Column(
          children: [
            Container(
              decoration: BoxDecoration(
                color: Theme.of(context).cardColor,
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.04),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Column(
                children: [
                  _buildSettingTile(
                    icon: Icons.notifications_none_rounded,
                    title: 'Notifikasi',
                    onTap: () {},
                  ),
                  const Divider(
                    height: 1,
                    color: Color(0xFFF1F5F9),
                    indent: 56,
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 6,
                    ),
                    child: Row(
                      children: [
                        Container(
                          width: 40,
                          height: 40,
                          decoration: BoxDecoration(
                            color: isDark
                                ? const Color(0xFF334155)
                                : const Color(0xFFF8FAFC),
                            shape: BoxShape.circle,
                          ),
                          child: Icon(
                            Icons.dark_mode_outlined,
                            color: isDark
                                ? const Color(0xFF94A3B8)
                                : const Color(0xFF64748B),
                            size: 20,
                          ),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Text(
                            'Mode Gelap',
                            style: TextStyle(
                              color: isDark
                                  ? Colors.white
                                  : const Color(0xFF1E293B),
                              fontSize: 15,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                        Switch(
                          value: _isDarkModeEnabled,
                          activeThumbColor: const Color(0xFF236CFF),
                          onChanged: (val) {
                            setState(() {
                              _isDarkModeEnabled = val;
                            });
                            AppTheme.themeNotifier.value = val
                                ? ThemeMode.dark
                                : ThemeMode.light;
                          },
                        ),
                      ],
                    ),
                  ),
                  const Divider(
                    height: 1,
                    color: Color(0xFFF1F5F9),
                    indent: 56,
                  ),
                  _buildSettingTile(
                    icon: Icons.info_outline_rounded,
                    title: 'Tentang Aplikasi',
                    onTap: () => showAboutApp(context),
                  ),
                  const Divider(
                    height: 1,
                    color: Color(0xFFF1F5F9),
                    indent: 56,
                  ),
                  _buildSettingTile(
                    icon: Icons.logout_rounded,
                    title: 'Keluar',
                    iconColor: Colors.redAccent,
                    textColor: Colors.redAccent,
                    onTap: () => _handleLogout(context),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSettingTile({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
    Color? iconColor,
    Color? textColor,
  }) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return ListTile(
      leading: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: isDark ? const Color(0xFF334155) : const Color(0xFFF8FAFC),
          shape: BoxShape.circle,
        ),
        child: Icon(
          icon,
          color:
              iconColor ??
              (isDark ? const Color(0xFF94A3B8) : const Color(0xFF64748B)),
          size: 20,
        ),
      ),
      title: Text(
        title,
        style: TextStyle(
          color: textColor ?? (isDark ? Colors.white : const Color(0xFF1E293B)),
          fontSize: 15,
          fontWeight: FontWeight.w600,
        ),
      ),
      trailing: const Icon(
        Icons.chevron_right_rounded,
        color: Color(0xFF94A3B8),
        size: 20,
      ),
      onTap: onTap,
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
    );
  }
}
