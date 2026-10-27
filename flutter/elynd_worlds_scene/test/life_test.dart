// What lives in a world (shaders/living.frag; life.py decides which belong to it), drawn for real:
// each only where it belongs in the region map, and shaped as in nature:
//  * birds cross the sky only, never the grass or the water below it;
//  * a fish's rings and the raindrops only on the water, and a raindrop's ring flat, as seen at a
//    low angle (a grid scaled on the screen once stretched every ring into a tall oval);
//  * the painted stars twinkle, never the rim of the moon nor a light below the sky, and a shooting
//    star crosses the sky only;
//  * cloud shadows fall on the grass only.
// On a plain painting nothing else changes the picture, so a world with one of them is compared
// with the same world without it, at the same moment.
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:elynd_worlds_scene/elynd_worlds_scene.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

// As wide as the shader's reference (k = 1): a raindrop's ring is a few pixels across.
const _w = 540, _h = 960;

// The region map's bands: sky over the top third, grass down to 0.6, water below.
const _skyEnd = 320, _waterStart = 576;

// Stars painted in the night sky, lights painted on the land below it, and the moon.
const _stars = [
  Offset(60, 50),
  Offset(420, 90),
  Offset(150, 230),
  Offset(330, 270),
  Offset(480, 180),
];
const _lights = [Offset(100, 400), Offset(400, 450)];
const _moon = Offset(270, 140), _moonRadius = 36.0;

Future<Uint8List> _png(int w, int h, void Function(Canvas) paint) async {
  final recorder = ui.PictureRecorder();
  paint(Canvas(recorder));
  final image = await recorder.endRecording().toImage(w, h);
  return (await image.toByteData(
    format: ui.ImageByteFormat.png,
  ))!.buffer.asUint8List();
}

void _fill(Canvas canvas, Color color) => canvas.drawRect(
  Rect.fromLTWH(0, 0, _w.toDouble(), _h.toDouble()),
  Paint()..color = color,
);

/// Half size, as the engine writes it.
Future<Uint8List> _map() => _png(_w ~/ 2, _h ~/ 2, (canvas) {
  void band(int from, int to, Color color) => canvas.drawRect(
    Rect.fromLTRB(0, from / 2, _w / 2, to / 2),
    Paint()..color = color,
  );
  band(0, _skyEnd, const Color(0xFF00FF00));
  band(_skyEnd, _waterStart, const Color(0xFF0000FF));
  band(_waterStart, _h, const Color(0xFFFF0000));
});

Future<Uint8List> _night({bool stars = true}) => _png(_w, _h, (canvas) {
  _fill(canvas, const Color(0xFF18223A));
  if (!stars) return;
  for (final at in [..._stars, ..._lights]) {
    canvas.drawRect(
      Rect.fromCenter(center: at, width: 2, height: 2),
      Paint()..color = const Color(0xFFFFFFFF),
    );
  }
  canvas.drawCircle(
    _moon,
    _moonRadius,
    Paint()..color = const Color(0xFFEDEBDD),
  );
});

void main() {
  late MemoryImage map, grey, night, darkSky;

  setUpAll(() async {
    map = MemoryImage(await _map());
    grey = MemoryImage(
      await _png(_w, _h, (canvas) => _fill(canvas, const Color(0xFF808080))),
    );
    night = MemoryImage(await _night());
    darkSky = MemoryImage(await _night(stars: false));
  });
  // The shader is loaded once per app run; each test has its own clock.
  setUp(LivingBackground.debugReset);

  Future<Uint8List> shoot(
    WidgetTester tester,
    ImageProvider plate,
    WorldLiving living,
    double t,
  ) async {
    tester.view.physicalSize = const Size(_w + 0.0, _h + 0.0);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    final key = GlobalKey();
    await tester.pumpWidget(
      Directionality(
        textDirection: TextDirection.ltr,
        child: Align(
          alignment: Alignment.topLeft,
          child: RepaintBoundary(
            key: key,
            child: SizedBox(
              width: _w.toDouble(),
              height: _h.toDouble(),
              child: LivingBackground(
                plate: plate,
                living: living,
                clockSeconds: t,
              ),
            ),
          ),
        ),
      ),
    );
    for (var i = 0; i < 20; i++) {
      await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 5)),
      );
      await tester.pump(const Duration(milliseconds: 16));
    }
    expect(
      find.descendant(
        of: find.byType(LivingBackground),
        matching: find.byType(CustomPaint),
      ),
      findsOneWidget,
      reason: 'the shader is drawing (not the fallback image)',
    );
    final boundary =
        key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
    final image = (await tester.runAsync(() => boundary.toImage()))!;
    expect((image.width, image.height), (_w, _h));
    final bytes = await tester.runAsync(
      () => image.toByteData(format: ui.ImageByteFormat.rawRgba),
    );
    image.dispose();
    return bytes!.buffer.asUint8List();
  }

  /// The pixels where two shots differ.
  List<Offset> changed(Uint8List a, Uint8List b) => [
    for (var y = 0; y < _h; y++)
      for (var x = 0; x < _w; x++)
        if ((a[(y * _w + x) * 4] - b[(y * _w + x) * 4]).abs() +
                (a[(y * _w + x) * 4 + 1] - b[(y * _w + x) * 4 + 1]).abs() +
                (a[(y * _w + x) * 4 + 2] - b[(y * _w + x) * 4 + 2]).abs() >
            3)
          Offset(x.toDouble(), y.toDouble()),
  ];

  /// A world with all three regions; `grass` off keeps the grass still (no wind), so only what lives
  /// there changes the picture.
  WorldLiving world({
    bool grass = false,
    String birds = 'none',
    bool night = false,
    bool fish = false,
    bool shadows = false,
    String particles = 'none',
    double seed = 0.5,
  }) => WorldLiving(
    regions: map,
    water: true,
    sky: true,
    grass: grass,
    horizon: _waterStart / _h,
    birds: birds,
    night: night,
    fish: fish,
    shadows: shadows,
    particles: particles,
    seed: seed,
  );

  testWidgets('birds cross the sky only: a flock, and gulls', (tester) async {
    // Seed 0.5: the first flock crosses from 7.8 s to 21.2 s, the first gulls from 7.8 s to 40.4 s.
    for (final (kind, t) in const [('flock', 14.5), ('gulls', 24.0)]) {
      final empty = await shoot(tester, grey, world(), t);
      final flying = await shoot(tester, grey, world(birds: kind), t);
      final diff = changed(empty, flying);
      expect(
        diff.where((p) => p.dy < _skyEnd).length,
        greaterThan(20),
        reason: '$kind in the sky',
      );
      expect(
        diff.where((p) => p.dy >= _skyEnd + 3),
        isEmpty,
        reason: 'no $kind below the sky',
      );
    }
  });

  testWidgets('a fish rises on the water only', (tester) async {
    // A fish rises at 0.2 s and at 17.3 s (the same in every world).
    var seen = 0;
    for (final t in const [0.6, 17.8]) {
      final diff = changed(
        await shoot(tester, grey, world(), t),
        await shoot(tester, grey, world(fish: true), t),
      );
      seen += diff.length;
      expect(
        diff.where((p) => p.dy < _waterStart - 3),
        isEmpty,
        reason: 'rings only on the water ($t s)',
      );
    }
    expect(seen, greaterThan(20), reason: 'the rings of a fish rising');
  });

  testWidgets(
    'raindrops ring on the water only, each ring flat as seen at a low angle',
    (tester) async {
      final calm = await shoot(tester, grey, world(), 3.0);
      final rain = await shoot(tester, grey, world(particles: 'rain'), 3.0);
      final diff = changed(calm, rain);
      expect(diff.length, greaterThan(500), reason: 'many drops at once');
      expect(
        diff.where((p) => p.dy < _waterStart - 3),
        isEmpty,
        reason: 'rings only on the water',
      );
      // A ring seen at a low angle is wide and flat, so the water changes faster up and down than
      // across (here 1.18 times; the tall ovals of a stretched grid made it 0.73).
      var across = 0, upDown = 0;
      int lum(int x, int y) =>
          rain[(y * _w + x) * 4] +
          rain[(y * _w + x) * 4 + 1] +
          rain[(y * _w + x) * 4 + 2];
      for (var y = _waterStart + 6; y < _h - 1; y++) {
        for (var x = 0; x < _w - 1; x++) {
          across += (lum(x + 1, y) - lum(x, y)).abs();
          upDown += (lum(x, y + 1) - lum(x, y)).abs();
        }
      }
      expect(
        upDown,
        greaterThan(across),
        reason: 'flat rings ($upDown up and down, $across across)',
      );
    },
  );

  testWidgets("the stars twinkle, never the moon's rim nor a light below the sky", (
    tester,
  ) async {
    // Before 5 s no shooting star can cross; the twinkle's rhythms are 1.7-4 s long.
    var twinkled = 0;
    for (final t in const [0.6, 1.5, 2.4, 3.3]) {
      final diff = changed(
        await shoot(tester, night, world(), t),
        await shoot(tester, night, world(night: true), t),
      );
      twinkled += diff.length;
      for (final p in diff) {
        // The sky drifts (here at most 10 px to the side): each change is at a star, wherever it is now.
        final atStar = _stars.any(
          (s) => (p.dx - s.dx).abs() <= 14 && (p.dy - s.dy).abs() <= 4,
        );
        expect(atStar, isTrue, reason: 'only stars twinkle, not $p ($t s)');
      }
    }
    expect(twinkled, greaterThan(0), reason: 'the stars twinkle');
  });

  testWidgets('a shooting star now and then, in the night sky only', (
    tester,
  ) async {
    // Seed 0.2: one starts at 28.25 s and lasts 0.8 s. A plain sky: nothing else to twinkle.
    final before = changed(
      await shoot(tester, darkSky, world(seed: 0.2), 27.0),
      await shoot(tester, darkSky, world(night: true, seed: 0.2), 27.0),
    );
    expect(before, isEmpty, reason: 'no shooting star yet');
    final diff = changed(
      await shoot(tester, darkSky, world(seed: 0.2), 28.65),
      await shoot(tester, darkSky, world(night: true, seed: 0.2), 28.65),
    );
    expect(diff.length, greaterThan(20), reason: 'a shooting star crossing');
    expect(
      diff.where((p) => p.dy >= _skyEnd + 3),
      isEmpty,
      reason: 'in the sky only',
    );
  });

  testWidgets('cloud shadows fall on the grass only', (tester) async {
    var shaded = 0;
    for (final t in const [0.0, 30.0, 60.0]) {
      final diff = changed(
        await shoot(tester, grey, world(grass: true), t),
        await shoot(tester, grey, world(grass: true, shadows: true), t),
      );
      shaded += diff.length;
      expect(
        diff.where((p) => p.dy < _skyEnd - 3 || p.dy >= _waterStart + 3),
        isEmpty,
        reason: 'shadows on the grass only ($t s)',
      );
    }
    expect(shaded, greaterThan(100), reason: "a cloud's shadow passing");
  });
}
