import 'package:flutter/material.dart';
import '../widgets/custom_button.dart';
import '../widgets/custom_textfield.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final usernameController = TextEditingController();
  final fullnameController = TextEditingController();
  final emailController = TextEditingController();
  final passwordController = TextEditingController();
  final confirmController = TextEditingController();

  bool obscure1 = true;
  bool obscure2 = true;

  void register() {
    if (usernameController.text.isEmpty ||
        fullnameController.text.isEmpty ||
        emailController.text.isEmpty ||
        passwordController.text.isEmpty ||
        confirmController.text.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Semua field wajib diisi')));
      return;
    }

    if (passwordController.text.length < 8) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Password minimal 8 karakter')),
      );
      return;
    }

    if (passwordController.text != confirmController.text) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Password tidak cocok')));
      return;
    }

    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Registrasi berhasil')));

    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Daftar Akun')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              CustomTextField(
                hint: 'Username',
                prefixIcon: Icons.person_outline,
                controller: usernameController,
              ),
              const SizedBox(height: 16),
              CustomTextField(
                hint: 'Nama Lengkap',
                prefixIcon: Icons.badge_outlined,
                controller: fullnameController,
              ),
              const SizedBox(height: 16),
              CustomTextField(
                hint: 'Email',
                prefixIcon: Icons.email_outlined,
                controller: emailController,
              ),
              const SizedBox(height: 16),
              CustomTextField(
                hint: 'Password',
                prefixIcon: Icons.lock_outline,
                controller: passwordController,
                obscureText: obscure1,
                suffixIcon: IconButton(
                  onPressed: () {
                    setState(() {
                      obscure1 = !obscure1;
                    });
                  },
                  icon: Icon(
                    obscure1 ? Icons.visibility_off : Icons.visibility,
                  ),
                ),
              ),
              const SizedBox(height: 16),
              CustomTextField(
                hint: 'Konfirmasi Password',
                prefixIcon: Icons.lock_outline,
                controller: confirmController,
                obscureText: obscure2,
                suffixIcon: IconButton(
                  onPressed: () {
                    setState(() {
                      obscure2 = !obscure2;
                    });
                  },
                  icon: Icon(
                    obscure2 ? Icons.visibility_off : Icons.visibility,
                  ),
                ),
              ),
              const SizedBox(height: 28),
              CustomButton(text: 'Daftar', onPressed: register),
            ],
          ),
        ),
      ),
    );
  }
}
