import 'dart:async';

import 'package:flutter/material.dart';
import 'package:risestock/screens/SecondScreen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    setState(() {});
    super.initState();
    Timer(const Duration(seconds: 3), () {
      Navigator.pushReplacement(context,
          MaterialPageRoute(builder: (context) => const SecondScreen()));
    });
  }

  @override
  Widget build(BuildContext context) {
    var deviceSize = MediaQuery.of(context).size;
    return Scaffold(
      body: Stack(
        children: [
          SizedBox(
            height: deviceSize.height,
            width: deviceSize.width,
            child: Image.asset(
              'assets/images/first_background.png',
              fit: BoxFit.fitHeight,
            ),
          ),
          Positioned(
              top: deviceSize.height / 1.7,
              child: Container(
                height: deviceSize.height / 2,
                width: deviceSize.width,
                decoration: const BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.only(
                        topLeft: Radius.circular(50),
                        topRight: Radius.circular(50))),
                child: Column(
                  children: [
                    SizedBox(
                      height: deviceSize.height / 20,
                      // height: 10,
                    ),
                    Image.asset(
                      'assets/images/home-trend-up.png',
                      scale: 1,
                    ),
                    SizedBox(
                      height: deviceSize.height / 20,
                    ),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: const [
                        Text(
                          'RISE',
                          style: TextStyle(color: Colors.blue, fontSize: 25),
                        ),
                        Text(
                          'STOCK',
                          style: TextStyle(color: Colors.black, fontSize: 25),
                        )
                      ],
                    )
                  ],
                ),
              ))
        ],
      ),
    );
  }
}
