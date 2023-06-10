import 'package:flutter/material.dart';
import 'package:risestock/screens/graph_screen.dart';

class CompanyScreen extends StatefulWidget {
  const CompanyScreen({super.key});

  @override
  State<CompanyScreen> createState() => _CompanyScreenState();
}

class _CompanyScreenState extends State<CompanyScreen> {
  @override
  Widget build(BuildContext context) {
    final currentWidth = MediaQuery.of(context).size.width;
    final currentHeight = MediaQuery.of(context).size.height;

    return SafeArea(
      child: Scaffold(
        body: SingleChildScrollView(
          child: Column(children: [
            Stack(
              children: [
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
              ],
            ),
            SizedBox(
              height: currentHeight * 0.1,
            ),
            Row(
              children: [
                const SizedBox(
                  width: 20,
                ),
                Text(
                  'Company Name',
                  style: TextStyle(
                    fontSize: currentHeight / 30,
                    fontWeight: FontWeight.w600,
                    color: const Color.fromARGB(255, 11, 49, 114),
                  ),
                ),
                const SizedBox(
                  width: 10,
                ),
                const Icon(Icons.home_filled)
              ],
            ),
            const SizedBox(
              height: 20,
            ),
            FractionallySizedBox(
              widthFactor: 0.8,
              child: Container(
                height: currentHeight * 0.08,
                decoration: const BoxDecoration(
                    color: Color.fromARGB(255, 100, 158, 185),
                    borderRadius: BorderRadius.all(Radius.circular(15))),
                child: Center(
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: TextFormField(
                      style: const TextStyle(color: Colors.white),
                      decoration: const InputDecoration(
                          isDense: true,
                          border: UnderlineInputBorder(
                              borderSide: BorderSide.none)),
                    ),
                  ),
                ),
              ),
            ),
            SizedBox(
              height: 20,
            ),
            Row(
              children: [
                const SizedBox(
                  width: 20,
                ),
                Text(
                  'Number Of Days : ',
                  style: TextStyle(
                    fontSize: currentHeight / 30,
                    fontWeight: FontWeight.w600,
                    color: const Color.fromARGB(255, 11, 49, 114),
                  ),
                ),
                const SizedBox(
                  width: 10,
                ),
                const Icon(Icons.calendar_today)
              ],
            ),
            SizedBox(
              height: 20,
            ),
            FractionallySizedBox(
              widthFactor: 0.8,
              child: Container(
                height: currentHeight * 0.08,
                decoration: const BoxDecoration(
                    color: Color.fromARGB(255, 100, 158, 185),
                    borderRadius: BorderRadius.all(Radius.circular(15))),
                child: Center(
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: TextFormField(
                      style: const TextStyle(color: Colors.white),
                      decoration: const InputDecoration(
                          isDense: true,
                          fillColor: Colors.white,
                          border: UnderlineInputBorder(
                              borderSide: BorderSide.none)),
                    ),
                  ),
                ),
              ),
            ),
            SizedBox(
              height: currentHeight * 0.1,
            ),
            Container(
              height: currentHeight * 0.05,
              width: currentWidth * 0.3,
              decoration: const BoxDecoration(
                  color: Color.fromARGB(255, 65, 179, 232),
                  borderRadius: BorderRadius.all(Radius.circular(20))),
              child: TextButton(
                  onPressed: () {
                    Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (context) => const GraphScreen()));
                  },
                  child: const Text(
                    "Predict",
                    style: TextStyle(
                      color: Color.fromARGB(255, 11, 49, 114),
                    ),
                  )),
            ),
          ]),
        ),
      ),
    );
  }
}
