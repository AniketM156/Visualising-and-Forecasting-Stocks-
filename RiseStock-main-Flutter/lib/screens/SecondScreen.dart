import 'package:flutter/material.dart';
import 'package:risestock/screens/loginScreen.dart';

class SecondScreen extends StatefulWidget {
  const SecondScreen({super.key});

  @override
  State<SecondScreen> createState() => _SecondScreenState();
}

class _SecondScreenState extends State<SecondScreen> {
  @override
  Widget build(BuildContext context) {
    var deviceSize = MediaQuery.of(context).size;

    return SafeArea(
      child: Scaffold(
        body: Column(children: [
          Stack(children: [
            Image.asset(
              'assets/images/image 6.png',
              fit: BoxFit.fitWidth,
            ),
            Positioned(
                top: 30,
                left: 15,
                child: Image.asset(
                  'assets/images/home-trend-up.png',
                  scale: 2,
                ))
          ]),
          Image.asset(
            'assets/images/image 1.png',
            fit: BoxFit.fitWidth,
          ),
          SizedBox(
            height: deviceSize.height / 25,
          ),
          const Text(
            'Visualizing and ForeCasting Stocks',
            style: TextStyle(
                fontWeight: FontWeight.w700,
                fontSize: 20,
                color: Color.fromARGB(255, 11, 49, 114)),
            maxLines: 2,
            textAlign: TextAlign.center,
          ),
          SizedBox(
            height: deviceSize.height / 25,
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
                          builder: (context) => const LoginScreen()));
                },
                child: const Text(
                  "Next",
                  style: TextStyle(
                    color: Color.fromARGB(255, 11, 49, 114),
                  ),
                )),
          ),
        ]),
      ),
    );
  }
}
