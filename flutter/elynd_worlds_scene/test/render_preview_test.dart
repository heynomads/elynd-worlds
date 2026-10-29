// Renders a saved world to PNG frames, to look at the motion without a phone:
//
//   flutter test test/render_preview_test.dart \
//     --dart-define=WORLD_DIR=/abs/out/<run>/<world> --dart-define=FRAMES_DIR=/abs/frames \
//     [--dart-define=SECONDS=8] [--dart-define=START=5] [--dart-define=CLOCK=0] [--dart-define=SILENT=true]
//     [--dart-define=RATIO=2] (pixels per point: 1.4 gives 546 x 1182, the review videos' width, about twice as fast)
//     [--dart-define=LIGHT=sunrise|deepen|nightfall] [--dart-define=DURATION=600] (the session, in seconds)
//     [--dart-define=TIMELAPSE=true] (the whole session's light within the clip, the world fully shown)
//     [--dart-define=SIZE=360x640] (the screen in points: 390x844 a phone, 360x640 the share card's 9:16)
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
// The world's own clock at the first frame, to start when a flock passes or a fish rises.
const _clock = int.fromEnvironment('CLOCK');
const _silent = bool.fromEnvironment('SILENT');
// LOOP=true: the shared clip's periods (every motion repeats every 12 s); a meditation's world never uses them.
const _loop = bool.fromEnvironment('LOOP');
final _ratio = double.tryParse(const String.fromEnvironment('RATIO')) ?? 2;
const _lightName = String.fromEnvironment('LIGHT');
const _duration = int.fromEnvironment('DURATION', defaultValue: 600);
final _size = (() {
  final wh = const String.fromEnvironment(
    'SIZE',
    defaultValue: '390x844',
  ).split('x');
  return Size(double.parse(wh[0]), double.parse(wh[1]));
})();
const _timelapse = bool.fromEnvironment('TIMELAPSE');

void main() {
  testWidgets('render a world to frames', (tester) async {
    if (_loop) WorldMotion.periods = WorldPeriods.clip;
    addTearDown(() => WorldMotion.periods = WorldPeriods.live);
    final manifest = WorldManifest.fromJson(
      jsonDecode(File('$_world/world.json').readAsStringSync()),
    )!;
    MemoryImage load(String name) =>
        MemoryImage(File('$_world/$name').readAsBytesSync());
    final background = load(
      manifest.moving ? manifest.plate! : manifest.painting,
    );
    final layers = manifest.moving
        ? [
            for (final l in manifest.layers)
              WorldLayer(
                image: load(l.source),
                depth: l.depth,
                motion: l.motion,
                attached: l.attached,
                glows: l.glows,
                anchor: l.anchor,
              ),
          ]
        : <WorldLayer>[];
    final living = manifest.living?.resolve((source) => load(source));

    tester.view.physicalSize = _size * 2;
    tester.view.devicePixelRatio = 2;
    addTearDown(tester.view.reset);
    final shot = GlobalKey();
    var elapsed = Duration.zero;
    // SILENT: as during a long silence of the voice.
    final timeline = WorldTimeline.build(
      durationSeconds: _duration.toDouble(),
      sleep: manifest.sleep,
      silences: _silent ? [WorldSilence(0, _duration.toDouble())] : const [],
    );
    await tester.pumpWidget(
      RepaintBoundary(
        key: shot,
        child: MediaQuery(
          data: MediaQueryData(size: _size),
          child: Directionality(
            textDirection: TextDirection.ltr,
            child: ColoredBox(
              color: const Color(0xFF152C2E),
              child: WorldScene(
                background: background,
                living: living,
                aspect: manifest.aspect,
                layers: layers,
                timeline: timeline,
                positionSeconds: () => _timelapse
                    ? _start +
                          (_duration - _start) *
                              elapsed.inMilliseconds /
                              1000 /
                              _seconds
                    : _start + elapsed.inMilliseconds / 1000,
                clockStartSeconds: _clock.toDouble(),
                followVisibility: !_timelapse,
                light: _lightName.isEmpty
                    ? LightJourney.none
                    : LightJourney.values.byName(_lightName),
              ),
            ),
          ),
        ),
      ),
    );
    await tester.runAsync(() async {
      final context = shot.currentContext!;
      for (final image in [
        background,
        ...layers.map((l) => l.image),
        ?living?.regions,
      ]) {
        await precacheImage(
          image,
          context,
        ).timeout(const Duration(seconds: 30));
      }
      // The living shader decodes its pictures itself: give it real time to.
      for (var i = 0; i < 20; i++) {
        await Future<void>.delayed(const Duration(milliseconds: 10));
      }
    });
    await tester.pump();
    // Sync: real file I/O never completes inside the test's fake clock.
    Directory(_frames).createSync(recursive: true);
    for (var i = 0; i < _seconds * 30; i++) {
      elapsed = Duration(microseconds: (i * 1e6 / 30).round());
      await tester.pump(const Duration(microseconds: 33333));
      await tester.runAsync(() async {
        final boundary =
            shot.currentContext!.findRenderObject()! as RenderRepaintBoundary;
        final image = await boundary.toImage(pixelRatio: _ratio);
        final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
        image.dispose();
        await File(
          '$_frames/${(i + 1).toString().padLeft(4, '0')}.png',
        ).writeAsBytes(bytes!.buffer.asUint8List());
      });
    }
  }, skip: _world.isEmpty || _frames.isEmpty);
}
