import 'package:flutter/material.dart';
import 'package:risestock/screens/company_name.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen>
    with SingleTickerProviderStateMixin {
  FocusNode _focusNode1 = FocusNode();
  FocusNode _focusNode2 = FocusNode();

  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this);
  }

  @override
  void dispose() {
    _focusNode1.dispose();
    _focusNode2.dispose();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    var deviceSize = MediaQuery.of(context).size;
    return GestureDetector(
      onTap: () {
        _focusNode1.unfocus();
        _focusNode2.unfocus();
      },
      child: SafeArea(
        child: Scaffold(
          body: Column(children: [
            SizedBox(
                child: Stack(
              children: [
                Image.asset(
                  'assets/images/image 6.png',
                  fit: BoxFit.fitWidth,
                ),
                Positioned(
                    top: 40,
                    left: 20,
                    child: Image.asset(
                      'assets/images/home-trend-up.png',
                      scale: 2,
                    ))
              ],
            )),
            const SizedBox(
              height: 35,
            ),
            const Text(
              'Please Login ',
              style: TextStyle(
                  color: Color.fromARGB(255, 11, 49, 114),
                  fontSize: 25,
                  fontWeight: FontWeight.w600),
            ),
            const SizedBox(
              height: 40,
            ),
            FractionallySizedBox(
              widthFactor: 0.8,
              child: TextFormField(
                focusNode: _focusNode1,
                decoration: InputDecoration(
                  prefixIcon: const Icon(Icons.person),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10.0),
                  ),
                  labelText: 'Username',
                ),
              ),
            ),
            const SizedBox(
              height: 20,
            ),
            FractionallySizedBox(
              widthFactor: 0.8,
              child: TextFormField(
                focusNode: _focusNode2,
                decoration: InputDecoration(
                  prefixIcon: const Icon(Icons.password_sharp),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10.0),
                  ),
                  labelText: 'Password',
                ),
              ),
            ),
            const SizedBox(
              height: 20,
            ),
            Container(
              height: 35,
              width: 80,
              decoration: const BoxDecoration(
                  color: Color.fromARGB(255, 65, 179, 232),
                  borderRadius: BorderRadius.all(Radius.circular(20))),
              child: TextButton(
                  onPressed: () {
                    Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (context) => const CompanyScreen()));
                  },
                  child: const Text(
                    "Login",
                    style: TextStyle(
                      color: Color.fromARGB(255, 11, 49, 114),
                    ),
                  )),
            ),
            const SizedBox(
              height: 20,
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text('Dont have a Account?'),
                InkWell(
                    onTap: () {},
                    child: const Text(
                      'Create One',
                      style:
                          TextStyle(color: Color.fromARGB(255, 65, 179, 232)),
                    ))
              ],
            )
          ]),
        ),
      ),
    );
  }
}
