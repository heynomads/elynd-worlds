// Renders a saved world to PNG frames, to look at the motion without a phone:
//
//   flutter test test/render_preview_test.dart \
//     --dart-define=WORLD_DIR=/abs/out/<run>/<world> --dart-define=FRAMES_DIR=/abs/frames \
//     [--dart-define=SECONDS=8] [--dart-define=START=5] [--dart-define=SILENT=true]
//   ffmpeg -framerate 30 -i /abs/frames/%04d.png -pix_fmt yuv420p preview.mp4
//
// Skipped unless WORLD_DIR and FRAMES_DIR are given.
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:elynd_worlds_scene/elynd_worlds_scene.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

const _world = String.fromEnvironment('WORLD_DIR');
const _frames = String.fromEnvironment('FRAMES_DIR');
const _seconds = int.fromEnvironment('SECONDS', defaultValue: 8);
const _start = int.fromEnvironment('START', defaultValue: 5);
const _silent = bool.fromEnvironment('SILENT');

void main() {
  testWidgets('render a world to frames', (tester) async {
    final manifest = WorldManifest.fromJson(jsonDecode(File('$_world/world.json').readAsStringSync()))!;
    MemoryImage load(String name) => MemoryImage(File('$_world/$name').readAsBytesSync());
    final background = load(manifest.moving ? manifest.plate! : manifest.painting);
    final layers = manifest.moving
        ? [for (final l in manifest.layers) WorldLayer(image: load(l.source), depth: l.depth, motion: l.motion, anchor: l.anchor)]
        : <WorldLayer>[];

    tester.view.physicalSize = const Size(780, 1688);
    tester.view.devicePixelRatio = 2;
    addTearDown(tester.view.reset);
    final shot = GlobalKey();
    var elapsed = Duration.zero;
    // SILENT: as during a long silence of the voice, where the breathing is deepest.
    final timeline = WorldTimeline.build(
        durationSeconds: 600, sleep: manifest.sleep, silences: _silent ? const [WorldSilence(0, 600)] : const []);
    await tester.pumpWidget(RepaintBoundary(
      key: shot,
      child: MediaQuery(
        data: const MediaQueryData(size: Size(390, 844)),
        child: Directionality(
          textDirection: TextDirection.ltr,
          child: ColoredBox(
            color: const Color(0xFF152C2E),
            child: WorldScene(
              background: background,
              aspect: manifest.aspect,
              layers: layers,
              timeline: timeline,
              positionSeconds: () => _start + elapsed.inMilliseconds / 1000,
            ),
          ),
        ),
      ),
    ));
    await tester.runAsync(() async {
      final context = shot.currentContext!;
      for (final image in [background, ...layers.map((l) => l.image)]) {
        await precacheImage(image, context).timeout(const Duration(seconds: 30));
      }
    });
    await tester.pump();
    // Sync: real file I/O never completes inside the test's fake clock.
    Directory(_frames).createSync(recursive: true);
    for (var i = 0; i < _seconds * 30; i++) {
      elapsed = Duration(microseconds: (i * 1e6 / 30).round());
      await tester.pump(const Duration(microseconds: 33333));
      await tester.runAsync(() async {
        final boundary = shot.currentContext!.findRenderObject()! as RenderRepaintBoundary;
        final image = await boundary.toImage(pixelRatio: 2);
        final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
        image.dispose();
        await File('$_frames/${(i + 1).toString().padLeft(4, '0')}.png').writeAsBytes(bytes!.buffer.asUint8List());
      });
    }
  }, skip: _world.isEmpty || _frames.isEmpty);
}
