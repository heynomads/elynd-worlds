import 'dart:math' as math;
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

    test('subjects painted low are lifted above the controls by a small zoom, at most 25%', () {
      const phone = Size(390, 844);
      final low = WorldScene.cover(phone, 9 / 16, (x: 0.5, y: 0.68));
      expect(low.height / 844, closeTo(1.1875, 1e-3)); // just enough: (1 - 0.62) / (1 - 0.68)
      expect(low.top + 0.68 * low.height, closeTo(0.62 * 844, 1e-6)); // the focus at 62% of the screen
      expect(low.bottom, greaterThanOrEqualTo(844)); // never an empty strip
      final veryLow = WorldScene.cover(phone, 9 / 16, (x: 0.5, y: 0.85));
      expect(veryLow.height / 844, closeTo(1.25, 1e-9)); // capped
      expect(WorldScene.cover(phone, 9 / 16, (x: 0.5, y: 0.5)).height, closeTo(844, 1e-9)); // no zoom
    });

    test('the manifest carries the focus', () {
      final m = WorldManifest.fromJson({'schema': 'elynd.world/1', 'focus': [0.7, 0.45], 'layers': []})!;
      expect(m.focus, (x: 0.7, y: 0.45));
      expect(WorldManifest.fromJson({'schema': 'elynd.world/1'})!.focus, (x: 0.5, y: 0.5));
    });
  });

  group('motion', () {
    // A boat that grew and shrank with each breath moved like a heartbeat. So: nothing grows or
    // shrinks, still things stay still, and each object moves only as such a thing moves, slowly.
    const kinds = [('float', 'ground'), ('float', 'air'), ('sway', 'edge'), ('drift', 'air'), ('still', 'ground')];

    test('still things stay still, always', () {
      for (var t = 0.0; t < 120; t += 0.7) {
        final pose = WorldMotion.layer(motion: 'still', clockSeconds: t, index: 1);
        expect((pose.dx, pose.dy, pose.rotation), (0.0, 0.0, 0.0));
      }
    });

    test('a boat on the water rises, falls and rolls a little; a butterfly loops; a branch sways', () {
      var bob = 0.0, roll = 0.0, loop = 0.0, sway = 0.0;
      for (var t = 0.0; t < 120; t += 0.25) {
        final boat = WorldMotion.layer(motion: 'float', attached: 'ground', clockSeconds: t);
        final butterfly = WorldMotion.layer(motion: 'float', attached: 'air', clockSeconds: t);
        final branch = WorldMotion.layer(motion: 'sway', attached: 'edge', clockSeconds: t);
        expect(boat.dx, 0);
        expect(boat.dy.abs(), lessThanOrEqualTo(WorldMotion.bob));
        expect(boat.rotation.abs(), lessThanOrEqualTo(WorldMotion.bobRoll));
        expect(butterfly.rotation, 0);
        expect(butterfly.dx.abs(), lessThanOrEqualTo(WorldMotion.hover));
        expect(butterfly.dy.abs(), lessThanOrEqualTo(WorldMotion.hover));
        expect((branch.dx, branch.dy), (0.0, 0.0));
        expect(branch.rotation.abs(), lessThanOrEqualTo(WorldMotion.sway));
        bob = math.max(bob, boat.dy.abs());
        roll = math.max(roll, boat.rotation.abs());
        loop = math.max(loop, butterfly.dx.abs());
        sway = math.max(sway, branch.rotation.abs());
      }
      // ...and they do move: a world is alive, just calm.
      expect(bob, greaterThan(WorldMotion.bob * 0.5));
      expect(roll, greaterThan(WorldMotion.bobRoll * 0.5));
      expect(loop, greaterThan(WorldMotion.hover * 0.5));
      expect(sway, greaterThan(WorldMotion.sway * 0.5));
    });

    test('slowly: from one frame to the next nothing moves more than a hair', () {
      for (var t = 0.0; t < 120; t += 0.37) {
        for (final (motion, attached) in kinds) {
          final a = WorldMotion.layer(motion: motion, attached: attached, clockSeconds: t, index: 2);
          final b = WorldMotion.layer(motion: motion, attached: attached, clockSeconds: t + 1 / 60, index: 2);
          // 0.00006 of the painting's height is about 0.05 pt on a phone.
          expect((a.dx - b.dx).abs(), lessThan(0.00006), reason: '$motion/$attached dx at $t');
          expect((a.dy - b.dy).abs(), lessThan(0.00006), reason: '$motion/$attached dy at $t');
          expect((a.rotation - b.rotation).abs(), lessThan(0.0003), reason: '$motion/$attached at $t');
        }
        final c = WorldMotion.camera(t), d = WorldMotion.camera(t + 1 / 60);
        expect((c.dx - d.dx).abs(), lessThan(0.00006));
        expect((c.dy - d.dy).abs(), lessThan(0.00006));
      }
    });

    test('the camera drifts a little, slowly, and near layers drift a little more (depth)', () {
      for (var t = 0.0; t < 200; t += 1.3) {
        final c = WorldMotion.camera(t);
        expect(c.dx.abs(), lessThanOrEqualTo(WorldMotion.cameraDrift));
        expect(c.dy.abs(), lessThanOrEqualTo(WorldMotion.cameraDrift));
      }
      expect(WorldMotion.parallax('near'), greaterThan(WorldMotion.parallax('mid')));
      expect(WorldMotion.parallax('mid'), greaterThan(1));
      // The margin covers the furthest drift of a near layer, so no edge ever shows.
      expect((WorldMotion.cameraHeadroom - 1) / 2, greaterThan(WorldMotion.cameraDrift * WorldMotion.parallax('near')));
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

    List<Matrix4> transforms(WidgetTester tester) =>
        [for (final t in tester.widgetList<Transform>(find.byType(Transform))) t.transform];

    testWidgets('nothing ever grows or shrinks: a constant margin, layers only move and turn', (tester) async {
      await tester.pumpWidget(host(WorldScene(
        background: MemoryImage(_png),
        aspect: 9 / 16,
        layers: [
          WorldLayer(image: MemoryImage(_png), depth: 'near', motion: 'float', attached: 'ground'),
          WorldLayer(image: MemoryImage(_png), motion: 'sway', attached: 'edge', anchor: (x: 0.0, y: 0.5)),
        ],
        timeline: const WorldTimeline(breath: [0, 1, 0, 1], visibility: [1, 1, 1, 1]),
        positionSeconds: () => 0,
      )));
      for (var i = 0; i < 12; i++) {
        await tester.pump(const Duration(milliseconds: 700));
        final [margin, ...layers] = transforms(tester);
        expect(margin.getMaxScaleOnAxis(), closeTo(WorldMotion.cameraHeadroom, 1e-9));
        for (final m in layers) {
          expect(m.getMaxScaleOnAxis(), closeTo(1.0, 1e-9));
        }
      }
    });

    testWidgets('reduce motion: nothing moves at all', (tester) async {
      await tester.pumpWidget(host(WorldScene(
        background: MemoryImage(_png),
        aspect: 9 / 16,
        layers: [WorldLayer(image: MemoryImage(_png), depth: 'near', motion: 'sway')],
        timeline: const WorldTimeline(breath: [1, 1], visibility: [1, 1]),
        positionSeconds: () => 0,
      ), reduceMotion: true));
      await tester.pump(const Duration(seconds: 2));
      final layer = transforms(tester).last;
      expect(layer.getTranslation().x, 0);
      expect(layer.getTranslation().y, 0);
      expect(layer.getMaxScaleOnAxis(), closeTo(1.0, 1e-9));
      expect(layer.storage[1], closeTo(0, 1e-12)); // no rotation
      // The plate's box holds an Image, a layer's a Transform (the test images'
      // error placeholders add boxes of their own).
      final boxes = tester.widgetList<Positioned>(find.byType(Positioned));
      final plate = boxes.firstWhere((p) => p.child is Image);
      final layerBox = boxes.firstWhere((p) => p.child is Transform);
      expect((plate.left, plate.top), (layerBox.left, layerBox.top)); // the camera stays put
    });
  });
}
