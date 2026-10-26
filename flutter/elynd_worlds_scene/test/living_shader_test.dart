// The living painting's shader (shaders/living.frag), drawn for real:
//  * with motion off it IS the plate, pixel for pixel, wherever the painting
//    sits on the screen (its coordinates are right);
//  * with motion on, the water and the sky move and the land between them
//    stays exactly as painted.
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:elynd_worlds_scene/elynd_worlds_scene.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

const _w = 90, _h = 160;

/// A painting with detail everywhere (so any shift shows): stripes and a gradient.
Future<Uint8List> _plate() async {
  final recorder = ui.PictureRecorder();
  final canvas = Canvas(recorder);
  for (var y = 0; y < _h; y++) {
    for (var x = 0; x < _w; x += 6) {
      final v = (x * 7 + y * 3) % 255;
      canvas.drawRect(
        Rect.fromLTWH(x.toDouble(), y.toDouble(), 3, 1),
        Paint()
          ..color = Color.fromARGB(255, v, (y * 255 / _h).round(), 255 - v),
      );
    }
  }
  final image = await recorder.endRecording().toImage(_w, _h);
  return (await image.toByteData(
    format: ui.ImageByteFormat.png,
  ))!.buffer.asUint8List();
}

/// A pale sky with a dark treetop in it (x 40..50, y 20..50), and its region map
/// (half size, as the engine writes it): sky over the top third, but not the treetop.
Future<(Uint8List, Uint8List)> _treetop() async {
  Future<Uint8List> png(int w, int h, void Function(Canvas) paint) async {
    final recorder = ui.PictureRecorder();
    paint(Canvas(recorder));
    final image = await recorder.endRecording().toImage(w, h);
    return (await image.toByteData(
      format: ui.ImageByteFormat.png,
    ))!.buffer.asUint8List();
  }

  final painting = await png(_w, _h, (canvas) {
    canvas.drawRect(
      Rect.fromLTWH(0, 0, _w.toDouble(), _h.toDouble()),
      Paint()..color = const Color(0xFFC8C8C8),
    );
    canvas.drawRect(
      const Rect.fromLTWH(40, 20, 10, 30),
      Paint()..color = const Color(0xFF141414),
    );
  });
  final map = await png(_w ~/ 2, _h ~/ 2, (canvas) {
    canvas.drawRect(
      Rect.fromLTWH(0, 0, _w / 2, _h / 2),
      Paint()..color = const Color(0xFF000000),
    );
    canvas.drawRect(
      Rect.fromLTWH(0, 0, _w / 2, _h / 6),
      Paint()..color = const Color(0xFF00FF00),
    );
    canvas.drawRect(
      const Rect.fromLTWH(20, 10, 5, 15),
      Paint()..color = const Color(0xFF000000),
    );
  });
  return (painting, map);
}

void main() {
  late MemoryImage plate, regions;

  late MemoryImage streamRegions;

  setUpAll(() async {
    plate = MemoryImage(await _plate());
    // Written exactly as elynd_worlds/regions.py writes a map (PNG, straight alpha, alpha =
    // 1 - stream): the phone keeps it premultiplied, which once wiped the water and the sky.
    regions = MemoryImage(
      File('test/fixtures/regions_lake.png').readAsBytesSync(),
    );
    streamRegions = MemoryImage(
      File('test/fixtures/regions_stream.png').readAsBytesSync(),
    );
  });
  // The shader is loaded once per app run; each test has its own clock.
  setUp(LivingBackground.debugReset);

  Finder drawing() => find.descendant(
    of: find.byType(LivingBackground),
    matching: find.byType(CustomPaint),
  );

  Future<Uint8List> shoot(
    WidgetTester tester,
    Widget child, {
    Offset at = Offset.zero,
  }) async {
    final key = GlobalKey();
    await tester.pumpWidget(
      Directionality(
        textDirection: TextDirection.ltr,
        // Loose constraints: the snapshot is exactly the painting's box, not the test's screen.
        child: Align(
          alignment: Alignment.topLeft,
          child: RepaintBoundary(
            key: key,
            child: SizedBox(
              width: _w + at.dx,
              height: _h + at.dy,
              child: Stack(
                children: [
                  Positioned(
                    left: at.dx,
                    top: at.dy,
                    width: _w.toDouble(),
                    height: _h.toDouble(),
                    child: child,
                  ),
                ],
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
    final boundary =
        key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
    final image = await tester.runAsync(() => boundary.toImage());
    expect(
      (image!.width, image.height),
      (_w + at.dx.toInt(), _h + at.dy.toInt()),
    );
    final bytes = await tester.runAsync(
      () => image.toByteData(format: ui.ImageByteFormat.rawRgba),
    );
    return bytes!.buffer.asUint8List();
  }

  const living = WorldLiving(water: true, sky: true, horizon: 0.6);

  Widget shader(double t, {bool still = false, bool withStream = false}) =>
      LivingBackground(
        plate: plate,
        living: WorldLiving(
          regions: withStream ? streamRegions : regions,
          water: living.water,
          sky: living.sky,
          stream: withStream,
          horizon: living.horizon,
        ),
        clockSeconds: t,
        still: still,
      );

  int differing(Uint8List a, Uint8List b, int width, int y0, int y1) {
    var n = 0;
    for (var y = y0; y < y1; y++) {
      for (var x = 0; x < width; x++) {
        final i = (y * width + x) * 4;
        if ((a[i] - b[i]).abs() +
                (a[i + 1] - b[i + 1]).abs() +
                (a[i + 2] - b[i + 2]).abs() >
            24) {
          n++;
        }
      }
    }
    return n;
  }

  testWidgets(
    'motion off: the shader draws the plate exactly, wherever the painting sits',
    (tester) async {
      for (final at in const [Offset.zero, Offset(37, 23)]) {
        final painted = await shoot(
          tester,
          Image(image: plate, fit: BoxFit.fill),
          at: at,
        );
        expect(find.byType(CustomPaint), findsNothing);
        final shaded = await shoot(tester, shader(5.0, still: true), at: at);
        expect(
          drawing(),
          findsOneWidget,
          reason: 'the shader is drawing (not the fallback image)',
        );
        final width = _w + at.dx.toInt();
        expect(
          differing(painted, shaded, width, 0, _h + at.dy.toInt()),
          0,
          reason: 'offset $at',
        );
      }
    },
  );

  testWidgets(
    'motion on: the water and the sky move, the land between them stays as painted',
    (tester) async {
      final a = await shoot(tester, shader(0.0));
      expect(drawing(), findsOneWidget);
      final b = await shoot(tester, shader(1.3));
      expect(
        differing(a, b, _w, 0, (_h / 3).floor() - 2),
        greaterThan(0),
        reason: 'the sky drifts',
      );
      expect(
        differing(a, b, _w, (_h * 0.6).ceil() + 3, _h),
        greaterThan(_w),
        reason: 'the water ripples',
      );
      expect(
        differing(a, b, _w, (_h / 3).ceil() + 3, (_h * 0.6).floor() - 3),
        0,
        reason: 'the land is still',
      );
    },
  );

  testWidgets('only sky drifts into the sky: a treetop beside it never comes along', (
    tester,
  ) async {
    final (painting, map) = (await tester.runAsync(_treetop))!;
    final scene = LivingBackground(
      plate: MemoryImage(painting),
      living: WorldLiving(regions: MemoryImage(map), sky: true),
      clockSeconds: 22.5, // the drift at its widest
    );
    final shot = await shoot(tester, scene);
    expect(drawing(), findsOneWidget);
    var dark = 0;
    for (var y = 25; y < 45; y++) {
      for (var x = 52; x < 56; x++) {
        if (shot[(y * _w + x) * 4] < 150) dark++;
      }
    }
    expect(dark, 0, reason: 'no ghost of the treetop in the sky beside it');
  });

  testWidgets(
    'a stream flows, and the lake and the sky in the same map still move',
    (tester) async {
      final a = await shoot(tester, shader(0.0, withStream: true));
      expect(drawing(), findsOneWidget);
      // 1.3 s: the sky drifts slowly (a third of a pixel at 0.7 s in this small painting).
      final b = await shoot(tester, shader(1.3, withStream: true));
      // The stream's band: map rows 30..40 = painting rows 60..80.
      expect(
        differing(a, b, _w, 62, 78),
        greaterThan(_w),
        reason: 'the stream flows',
      );
      expect(
        differing(a, b, _w, 0, (_h / 3).floor() - 2),
        greaterThan(0),
        reason: 'the sky still drifts',
      );
      expect(
        differing(a, b, _w, (_h * 0.6).ceil() + 3, _h),
        greaterThan(_w),
        reason: 'the water still ripples',
      );
      expect(
        differing(a, b, _w, (_h / 3).ceil() + 3, 57),
        0,
        reason: 'the land above the stream is still',
      );
    },
  );
}
