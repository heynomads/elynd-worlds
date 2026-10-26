import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:elynd_worlds_scene/elynd_worlds_scene.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

// A 1x1 transparent PNG.
final Uint8List _png = Uint8List.fromList(const [
  137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31,
  21, 196, 137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13, 10, 45, 180, 0, 0, 0,
  0, 73, 69, 78, 68, 174, 66, 96, 130,
]);

void main() {
  group('timeline', () {
    final cases = (jsonDecode(File('test/fixtures/timeline_parity.json').readAsStringSync()) as List)
        .cast<Map<String, dynamic>>();
    for (final c in cases) {
      test('frame for frame the same as Python: ${c['duration']} s, sleep ${c['sleep']}', () {
        final tl = WorldTimeline.build(
          durationSeconds: (c['duration'] as num).toDouble(),
          silences: [for (final s in c['silences'] as List) WorldSilence((s[0] as num).toDouble(), (s[1] as num).toDouble())],
          cues: [for (final q in c['cues'] as List) WorldCue((q[1] as num).toDouble(), inhale: q[0] == 'inhale')],
          sleep: c['sleep'] as bool,
        );
        final breath = (c['breath'] as List).cast<num>();
        final visibility = (c['visibility'] as List).cast<num>();
        expect(tl.breath.length, breath.length);
        for (var i = 0; i < breath.length; i++) {
          expect(tl.breath[i], closeTo(breath[i], 0.0011), reason: 'breath at $i s');
          expect(tl.visibility[i], closeTo(visibility[i], 0.0011), reason: 'visibility at $i s');
        }
      });
    }

    test('interpolates between seconds and holds at the ends', () {
      const tl = WorldTimeline(breath: [0, 1, 0], visibility: [1, 0.5, 0.25]);
      expect(tl.breathAt(0.5), closeTo(0.5, 1e-9));
      expect(tl.visibilityAt(-3), 1.0);
      expect(tl.visibilityAt(99), 0.25);
    });

    test('silence regions parse leniently', () {
      expect(WorldSilence.fromJson({'start_seconds': 10, 'end_seconds': 40.5})!.end, 40.5);
      expect(WorldSilence.fromJson({'start_seconds': 40, 'end_seconds': 10}), isNull);
      expect(WorldSilence.fromJson('x'), isNull);
    });
  });

  group('manifest', () {
    test('reads world.json and the server block, refusing what it cannot use', () {
      final m = WorldManifest.fromJson({
        'schema': 'elynd.world/1', 'title': 'Still Lake', 'sleep': false, 'size': [1152, 2048],
        'painting': 'painting.png', 'plate': 'plate.png',
        'layers': [
          {'file': 'layer_0.png', 'depth': 'near', 'motion': 'still', 'anchor': [0.3, 0.9]},
          {'url': 'https://x/l1.webp', 'depth': 'far?', 'motion': 'spin', 'anchor': 'nope'},
          {'depth': 'near'},
        ],
      })!;
      expect(m.aspect, closeTo(0.5625, 1e-9));
      expect(m.moving, isTrue);
      expect(m.layers.map((l) => l.source), ['layer_0.png', 'https://x/l1.webp']);
      expect((m.layers[1].depth, m.layers[1].motion, m.layers[1].anchor), ('mid', 'still', (x: 0.5, y: 1.0)));
      expect(WorldManifest.fromJson({'schema': 'other'}), isNull);
    });
  });

  group('focus', () {
    test('a phone crops a painting around its subjects, never past its edges', () {
      const phone = Size(390, 844);
      final centred = WorldScene.cover(phone, 9 / 16, (x: 0.5, y: 0.5));
      expect(centred.height, closeTo(844, 1e-9));
      expect(centred.center.dx, closeTo(195, 1e-9));
      final right = WorldScene.cover(phone, 9 / 16, (x: 0.8, y: 0.5));
      expect(right.right, closeTo(390, 1e-9)); // as far as the edge allows
      final nearRight = WorldScene.cover(phone, 9 / 16, (x: 0.55, y: 0.5));
      expect(0.55 * nearRight.width + nearRight.left, closeTo(195, 1e-6)); // the focus in the middle
      expect(WorldScene.cover(const Size(1000, 500), 9 / 16, (x: 0.5, y: 0.3)).top, lessThanOrEqualTo(0));
    });

    test('the manifest carries the focus', () {
      final m = WorldManifest.fromJson({'schema': 'elynd.world/1', 'focus': [0.7, 0.45], 'layers': []})!;
      expect(m.focus, (x: 0.7, y: 0.45));
      expect(WorldManifest.fromJson({'schema': 'elynd.world/1'})!.focus, (x: 0.5, y: 0.5));
    });
  });

  group('motion', () {
    test('breathing only ever grows a layer about its anchor (never uncovers the plate)', () {
      for (var b = 0.0; b <= 1.0; b += 0.25) {
        final pose = WorldMotion.layer(depth: 'near', motion: 'still', breath: b, clockSeconds: b * 7);
        expect(pose.scale, greaterThanOrEqualTo(1.0));
        expect(pose.rotation, 0);
      }
      expect(WorldMotion.layer(depth: 'near', motion: 'still', breath: 1, clockSeconds: 0).scale,
          greaterThan(WorldMotion.layer(depth: 'mid', motion: 'still', breath: 1, clockSeconds: 0).scale));
    });

    test('sway rotates, float lifts, still does neither; all stay small', () {
      final sway = WorldMotion.layer(depth: 'near', motion: 'sway', breath: 0.5, clockSeconds: 2.25);
      final float = WorldMotion.layer(depth: 'near', motion: 'float', breath: 0.5, clockSeconds: 1.75);
      expect(sway.rotation.abs(), greaterThan(0));
      expect(sway.rotation.abs(), lessThanOrEqualTo(WorldMotion.swayRadians));
      expect(float.dy.abs(), greaterThan(0));
      expect(float.dy.abs(), lessThanOrEqualTo(WorldMotion.float));
    });
  });

  group('scene', () {
    Widget host(WorldScene scene, {bool reduceMotion = false}) => MediaQuery(
          data: MediaQueryData(disableAnimations: reduceMotion, size: const Size(390, 844)),
          child: Directionality(textDirection: TextDirection.ltr, child: scene),
        );

    testWidgets('draws the plate and every layer, and fades with the timeline', (tester) async {
      var seconds = 0.0;
      await tester.pumpWidget(host(WorldScene(
        background: MemoryImage(_png),
        aspect: 9 / 16,
        layers: [
          WorldLayer(image: MemoryImage(_png), depth: 'near', motion: 'sway', anchor: (x: 0.0, y: 0.2)),
          WorldLayer(image: MemoryImage(_png), anchor: (x: 0.5, y: 0.9)),
        ],
        timeline: const WorldTimeline(breath: [0, 1, 0], visibility: [1, 0.25, 0.25]),
        positionSeconds: () => seconds,
      )));
      expect(find.byType(Image), findsNWidgets(3));
      expect(tester.widget<Opacity>(find.byType(Opacity)).opacity, 1.0);
      seconds = 2;
      await tester.pump(const Duration(milliseconds: 16));
      expect(tester.widget<Opacity>(find.byType(Opacity)).opacity, closeTo(0.25, 0.01));
    });

    testWidgets('a paused session stops breathing ahead after one second', (tester) async {
      await tester.pumpWidget(host(WorldScene(
        background: MemoryImage(_png),
        aspect: 9 / 16,
        timeline: const WorldTimeline(breath: [0, 0, 0, 0], visibility: [1, 1, 0.5, 0]),
        positionSeconds: () => 0,
      )));
      await tester.pump(const Duration(seconds: 3));
      expect(tester.widget<Opacity>(find.byType(Opacity)).opacity, closeTo(1.0, 0.01));
    });

    testWidgets('reduce motion: no layer moves', (tester) async {
      await tester.pumpWidget(host(WorldScene(
        background: MemoryImage(_png),
        aspect: 9 / 16,
        layers: [WorldLayer(image: MemoryImage(_png), depth: 'near', motion: 'sway')],
        timeline: const WorldTimeline(breath: [1, 1], visibility: [1, 1]),
        positionSeconds: () => 0,
      ), reduceMotion: true));
      await tester.pump(const Duration(seconds: 2));
      final transform = tester.widgetList<Transform>(find.byType(Transform)).last.transform;
      expect(transform.getMaxScaleOnAxis(), closeTo(1.0, 1e-9));
    });
  });
}
