import 'dart:math' as math;
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:elynd_worlds_scene/elynd_worlds_scene.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

// A 1x1 transparent PNG.
final Uint8List _png = Uint8List.fromList(const [
  137,
  80,
  78,
  71,
  13,
  10,
  26,
  10,
  0,
  0,
  0,
  13,
  73,
  72,
  68,
  82,
  0,
  0,
  0,
  1,
  0,
  0,
  0,
  1,
  8,
  6,
  0,
  0,
  0,
  31,
  21,
  196,
  137,
  0,
  0,
  0,
  13,
  73,
  68,
  65,
  84,
  120,
  156,
  99,
  0,
  1,
  0,
  0,
  5,
  0,
  1,
  13,
  10,
  45,
  180,
  0,
  0,
  0,
  0,
  73,
  69,
  78,
  68,
  174,
  66,
  96,
  130,
]);

void main() {
  group('timeline', () {
    final cases =
        (jsonDecode(
                  File('test/fixtures/timeline_parity.json').readAsStringSync(),
                )
                as List)
            .cast<Map<String, dynamic>>();
    for (final c in cases) {
      test(
        'frame for frame the same as Python: ${c['duration']} s, sleep ${c['sleep']}',
        () {
          final tl = WorldTimeline.build(
            durationSeconds: (c['duration'] as num).toDouble(),
            silences: [
              for (final s in c['silences'] as List)
                WorldSilence(
                  (s[0] as num).toDouble(),
                  (s[1] as num).toDouble(),
                ),
            ],
            cues: [
              for (final q in c['cues'] as List)
                WorldCue((q[1] as num).toDouble(), inhale: q[0] == 'inhale'),
            ],
            sleep: c['sleep'] as bool,
          );
          final breath = (c['breath'] as List).cast<num>();
          final visibility = (c['visibility'] as List).cast<num>();
          expect(tl.breath.length, breath.length);
          for (var i = 0; i < breath.length; i++) {
            expect(
              tl.breath[i],
              closeTo(breath[i], 0.0011),
              reason: 'breath at $i s',
            );
            expect(
              tl.visibility[i],
              closeTo(visibility[i], 0.0011),
              reason: 'visibility at $i s',
            );
          }
        },
      );
    }

    test('interpolates between seconds and holds at the ends', () {
      const tl = WorldTimeline(breath: [0, 1, 0], visibility: [1, 0.5, 0.25]);
      expect(tl.breathAt(0.5), closeTo(0.5, 1e-9));
      expect(tl.visibilityAt(-3), 1.0);
      expect(tl.visibilityAt(99), 0.25);
    });

    test('silence regions parse leniently', () {
      expect(
        WorldSilence.fromJson({'start_seconds': 10, 'end_seconds': 40.5})!.end,
        40.5,
      );
      expect(
        WorldSilence.fromJson({'start_seconds': 40, 'end_seconds': 10}),
        isNull,
      );
      expect(WorldSilence.fromJson('x'), isNull);
    });
  });

  group('manifest', () {
    test(
      'reads world.json and the server block, refusing what it cannot use',
      () {
        final m = WorldManifest.fromJson({
          'schema': 'elynd.world/1',
          'title': 'Still Lake',
          'sleep': false,
          'size': [1152, 2048],
          'painting': 'painting.png',
          'plate': 'plate.png',
          'layers': [
            {
              'file': 'layer_0.png',
              'depth': 'near',
              'motion': 'still',
              'anchor': [0.3, 0.9],
            },
            {
              'url': 'https://x/l1.webp',
              'depth': 'far?',
              'motion': 'spin',
              'anchor': 'nope',
            },
            {'depth': 'near'},
          ],
        })!;
        expect(m.aspect, closeTo(0.5625, 1e-9));
        expect(m.moving, isTrue);
        expect(m.layers.map((l) => l.source), [
          'layer_0.png',
          'https://x/l1.webp',
        ]);
        expect(
          (m.layers[1].depth, m.layers[1].motion, m.layers[1].anchor),
          ('mid', 'still', (x: 0.5, y: 1.0)),
        );
        expect(WorldManifest.fromJson({'schema': 'other'}), isNull);
      },
    );
  });

  group('focus', () {
    test(
      'a phone crops a painting around its subjects, never past its edges',
      () {
        const phone = Size(390, 844);
        final centred = WorldScene.cover(phone, 9 / 16, (x: 0.5, y: 0.5));
        expect(centred.height, closeTo(844, 1e-9));
        expect(centred.center.dx, closeTo(195, 1e-9));
        final right = WorldScene.cover(phone, 9 / 16, (x: 0.8, y: 0.5));
        expect(right.right, closeTo(390, 1e-9)); // as far as the edge allows
        final nearRight = WorldScene.cover(phone, 9 / 16, (x: 0.55, y: 0.5));
        expect(
          0.55 * nearRight.width + nearRight.left,
          closeTo(195, 1e-6),
        ); // the focus in the middle
        expect(
          WorldScene.cover(const Size(1000, 500), 9 / 16, (x: 0.5, y: 0.3)).top,
          lessThanOrEqualTo(0),
        );
      },
    );

    test(
      'subjects painted low are lifted above the controls by a small zoom, at most 25%',
      () {
        const phone = Size(390, 844);
        final low = WorldScene.cover(phone, 9 / 16, (x: 0.5, y: 0.68));
        expect(
          low.height / 844,
          closeTo(1.1875, 1e-3),
        ); // just enough: (1 - 0.62) / (1 - 0.68)
        expect(
          low.top + 0.68 * low.height,
          closeTo(0.62 * 844, 1e-6),
        ); // the focus at 62% of the screen
        expect(low.bottom, greaterThanOrEqualTo(844)); // never an empty strip
        final veryLow = WorldScene.cover(phone, 9 / 16, (x: 0.5, y: 0.85));
        expect(veryLow.height / 844, closeTo(1.25, 1e-9)); // capped
        expect(
          WorldScene.cover(phone, 9 / 16, (x: 0.5, y: 0.5)).height,
          closeTo(844, 1e-9),
        ); // no zoom
      },
    );

    test('the manifest carries the focus', () {
      final m = WorldManifest.fromJson({
        'schema': 'elynd.world/1',
        'focus': [0.7, 0.45],
        'layers': [],
      })!;
      expect(m.focus, (x: 0.7, y: 0.45));
      expect(WorldManifest.fromJson({'schema': 'elynd.world/1'})!.focus, (
        x: 0.5,
        y: 0.5,
      ));
    });
  });

  group('motion', () {
    // A growing and shrinking boat moved like a heartbeat, and a world that barely moved looked like a
    // static picture. So: objects move as such things move, clearly, and nothing pulses.
    const kinds = [
      ('float', 'ground'),
      ('float', 'air'),
      ('flutter', 'air'),
      ('sway', 'edge'),
      ('drift', 'air'),
    ];

    test('a meditation\'s world never repeats every 12 s: the live periods (29b9366) are the default', () {
      expect(WorldMotion.periods, same(WorldPeriods.live));
      const live = WorldPeriods.live;
      expect([live.swell, live.swellShort, live.roll, live.hoverX, live.hoverY, live.wing, live.sway, live.swayShort,
          live.drift], [4.3, 2.7, 5.1, 7.9, 5.3, 4.2, 4.6, 1.9, 14]);
      for (final (motion, attached) in kinds) {
        var differs = false;
        for (var t = 0.0; t < 24 && !differs; t += 0.37) {
          final a = WorldMotion.layer(motion: motion, attached: attached, clockSeconds: t);
          final b = WorldMotion.layer(motion: motion, attached: attached, clockSeconds: t + WorldMotion.loopSeconds);
          differs = (a.dx - b.dx).abs() + (a.dy - b.dy).abs() + (a.rotation - b.rotation).abs() > 1e-6;
        }
        expect(differs, isTrue, reason: '$motion/$attached would loop every 12 s in a meditation');
      }
    });

    test(
      'the shared clip\'s periods: every motion repeats every 12 s, so a 12-second clip loops without a seam',
      () {
        WorldMotion.periods = WorldPeriods.clip;
        addTearDown(() => WorldMotion.periods = WorldPeriods.live);
        for (final (motion, attached) in [...kinds, ('still', 'ground')]) {
          for (var index = 0; index < 3; index++) {
            for (var t = 0.0; t < 24; t += 0.37) {
              final a = WorldMotion.layer(
                motion: motion,
                attached: attached,
                clockSeconds: t,
                index: index,
              );
              final b = WorldMotion.layer(
                motion: motion,
                attached: attached,
                clockSeconds: t + WorldMotion.loopSeconds,
                index: index,
              );
              for (final (x, y) in [
                (a.dx, b.dx),
                (a.dy, b.dy),
                (a.rotation, b.rotation),
                (a.wings, b.wings),
              ]) {
                expect(y, closeTo(x, 1e-9), reason: '$motion/$attached at $t');
              }
            }
          }
        }
      },
    );

    test('still things stay still, always', () {
      for (var t = 0.0; t < 120; t += 0.7) {
        final pose = WorldMotion.layer(
          motion: 'still',
          clockSeconds: t,
          index: 1,
        );
        expect(
          (pose.dx, pose.dy, pose.rotation, pose.wings),
          (0.0, 0.0, 0.0, 1.0),
        );
      }
    });

    test(
      'a boat rides the ripples, a butterfly floats and folds its wings, a branch sways: clearly, within bounds',
      () {
        var bob = 0.0,
            roll = 0.0,
            loop = 0.0,
            sway = 0.0,
            folded = 0,
            frames = 0;
        var narrowest = 1.0, lowest = 1.0;
        for (var t = 0.0; t < 120; t += 1 / 30) {
          final boat = WorldMotion.layer(
            motion: 'float',
            attached: 'ground',
            clockSeconds: t,
          );
          final leaf = WorldMotion.layer(
            motion: 'float',
            attached: 'air',
            clockSeconds: t,
          );
          final butterfly = WorldMotion.layer(
            motion: 'flutter',
            attached: 'air',
            clockSeconds: t,
          );
          final branch = WorldMotion.layer(
            motion: 'sway',
            attached: 'edge',
            clockSeconds: t,
          );
          expect((boat.dx, boat.wings), (0.0, 1.0));
          // Lower than painted, never higher.
          expect(boat.dy, inInclusiveRange(-1e-12, WorldMotion.bob + 1e-12));
          expect(boat.rotation.abs(), lessThanOrEqualTo(WorldMotion.bobRoll));
          expect((leaf.rotation, leaf.wings), (0.0, 1.0));
          expect(leaf.dx.abs(), lessThanOrEqualTo(WorldMotion.hoverX));
          expect(leaf.dy.abs(), lessThanOrEqualTo(WorldMotion.hoverY));
          expect((branch.dx, branch.dy, branch.wings), (0.0, 0.0, 1.0));
          expect(branch.rotation.abs(), lessThanOrEqualTo(WorldMotion.sway));
          expect(
            butterfly.wings,
            inInclusiveRange(WorldMotion.wingsFolded - 1e-9, 1.0),
          );
          bob = math.max(bob, boat.dy);
          lowest = math.min(lowest, boat.dy);
          roll = math.max(roll, boat.rotation.abs());
          loop = math.max(loop, leaf.dx.abs());
          sway = math.max(sway, branch.rotation.abs());
          narrowest = math.min(narrowest, butterfly.wings);
          if (butterfly.wings < 0.95) folded++;
          frames++;
        }
        // They do move, enough to see.
        expect(bob - lowest, greaterThan(WorldMotion.bob * 0.6));
        expect(roll, greaterThan(WorldMotion.bobRoll * 0.6));
        expect(loop, greaterThan(WorldMotion.hoverX * 0.6));
        expect(sway, greaterThan(WorldMotion.sway * 0.6));
        // The wings fold, now and then, for a moment.
        expect(narrowest, lessThan(0.5));
        expect(folded / frames, inExclusiveRange(0.02, 0.2));
      },
    );

    test('calmly: from one frame to the next nothing jumps', () {
      for (var t = 0.0; t < 120; t += 0.37) {
        for (final (motion, attached) in kinds) {
          final a = WorldMotion.layer(
            motion: motion,
            attached: attached,
            clockSeconds: t,
            index: 2,
          );
          final b = WorldMotion.layer(
            motion: motion,
            attached: attached,
            clockSeconds: t + 1 / 60,
            index: 2,
          );
          // 0.0005 of the painting's height is under half a point on a phone; 0.003 rad is 0.17 degrees.
          expect(
            (a.dx - b.dx).abs(),
            lessThan(0.0005),
            reason: '$motion/$attached dx at $t',
          );
          expect(
            (a.dy - b.dy).abs(),
            lessThan(0.0005),
            reason: '$motion/$attached dy at $t',
          );
          expect(
            (a.rotation - b.rotation).abs(),
            lessThan(0.003),
            reason: '$motion/$attached at $t',
          );
        }
      }
    });

    test('nothing pulses: only a butterfly\'s wings ever change width', () {
      for (var t = 0.0; t < 60; t += 0.1) {
        for (final (motion, attached) in kinds) {
          if (motion == 'flutter') continue;
          expect(
            WorldMotion.layer(
              motion: motion,
              attached: attached,
              clockSeconds: t,
            ).wings,
            1.0,
          );
        }
      }
    });
  });

  group('living', () {
    test('the manifest and the server block both read "living"', () {
      final saved = LivingSpec.fromJson({
        'regions': 'regions.png',
        'water': true,
        'sky': true,
        'grass': false,
        'trees': true,
        'horizon': 0.52,
        'lights': [
          {'x': 0.5, 'y': 0.6, 'size': 0.08},
          'nope',
        ],
        'particles': 'fireflies',
      })!;
      expect(
        (
          saved.regions,
          saved.water,
          saved.sky,
          saved.grass,
          saved.stream,
          saved.trees,
          saved.horizon,
        ),
        ('regions.png', true, true, false, false, true, 0.52),
      );
      expect(saved.lights.single.size, 0.08);
      expect(saved.particles, 'fireflies');
      final sent = LivingSpec.fromJson({
        'regions_url': 'https://x/r.webp',
        'particles': 'lava',
      })!;
      expect((sent.regions, sent.particles), ('https://x/r.webp', 'none'));
      expect(LivingSpec.fromJson(null), isNull);
      final living = saved.resolve((source) => MemoryImage(_png));
      expect((living.mapped, living.shaded), (true, true));
      expect(const WorldLiving(trees: true).shaded, isTrue);
      expect(
        const WorldLiving(water: true).mapped,
        isFalse,
      ); // no map: nothing to read the water from
      final manifest = WorldManifest.fromJson({
        'schema': 'elynd.world/1',
        'living': {'trees': true},
      })!;
      expect(manifest.living!.trees, isTrue);
      expect(
        WorldManifest.fromJson({'schema': 'elynd.world/1'})!.living,
        isNull,
      );
    });
  });

  group('scene', () {
    Widget host(WorldScene scene, {bool reduceMotion = false}) => MediaQuery(
      data: MediaQueryData(
        disableAnimations: reduceMotion,
        size: const Size(390, 844),
      ),
      child: Directionality(textDirection: TextDirection.ltr, child: scene),
    );

    testWidgets(
      'draws the plate and every layer, and fades with the timeline',
      (tester) async {
        var seconds = 0.0;
        await tester.pumpWidget(
          host(
            WorldScene(
              background: MemoryImage(_png),
              aspect: 9 / 16,
              layers: [
                WorldLayer(
                  image: MemoryImage(_png),
                  depth: 'near',
                  motion: 'sway',
                  anchor: (x: 0.0, y: 0.2),
                ),
                WorldLayer(image: MemoryImage(_png), anchor: (x: 0.5, y: 0.9)),
              ],
              timeline: const WorldTimeline(
                breath: [0, 1, 0],
                visibility: [1, 0.25, 0.25],
              ),
              positionSeconds: () => seconds,
            ),
          ),
        );
        expect(find.byType(Image), findsNWidgets(3));
        expect(tester.widget<Opacity>(find.byType(Opacity)).opacity, 1.0);
        seconds = 2;
        await tester.pump(const Duration(milliseconds: 16));
        expect(
          tester.widget<Opacity>(find.byType(Opacity)).opacity,
          closeTo(0.25, 0.01),
        );
      },
    );

    testWidgets(
      'redrawn like a video: 30 times a second, 15 while only a glow, whatever the display',
      (tester) async {
        // The phone's battery: never redrawn faster than the motion needs.
        Future<int> redraws(List<double> visibility, Duration vsync) async {
          await tester.pumpWidget(
            host(
              WorldScene(
                key: UniqueKey(),
                background: MemoryImage(_png),
                aspect: 9 / 16,
                timeline: WorldTimeline(
                  breath: const [0, 0, 0, 0],
                  visibility: visibility,
                ),
                positionSeconds: () => 0,
              ),
            ),
          );
          await tester.pump(const Duration(milliseconds: 100));
          final before = WorldScene.debugRedraws;
          for (
            var t = Duration.zero;
            t < const Duration(seconds: 1);
            t += vsync
          ) {
            await tester.pump(vsync);
          }
          return WorldScene.debugRedraws - before;
        }

        const hz120 = Duration(microseconds: 8333),
            hz60 = Duration(microseconds: 16667);
        expect(
          await redraws(const [1, 1, 1, 1], hz120),
          inInclusiveRange(29, 31),
        );
        expect(
          await redraws(const [1, 1, 1, 1], hz60),
          inInclusiveRange(29, 31),
        );
        expect(
          await redraws(const [0.25, 0.25, 0.25, 0.25], hz120),
          inInclusiveRange(14, 16),
        );
      },
    );

    testWidgets(
      'and asks for no other frames: none between its redraws, whatever the display',
      (tester) async {
        // A running ticker asked for a frame at every vsync (60 or 120 a second), each drawn again on the GPU.
        Future<int> frames(List<double> visibility, Duration vsync) async {
          await tester.pumpWidget(
            host(
              WorldScene(
                key: UniqueKey(),
                background: MemoryImage(_png),
                aspect: 9 / 16,
                timeline: WorldTimeline(
                  breath: const [0, 0, 0, 0],
                  visibility: visibility,
                ),
                positionSeconds: () => 0,
              ),
            ),
          );
          await tester.pump(const Duration(milliseconds: 100));
          var asked = 0;
          for (
            var t = Duration.zero;
            t < const Duration(seconds: 1);
            t += vsync
          ) {
            // Time passes until the vsync (its timers run), then the frame: an engine draws one only when asked,
            // and the scene asks while its ticker waits.
            await tester.binding.delayed(vsync);
            if (tester.binding.transientCallbackCount > 0) asked++;
            await tester.pump();
          }
          return asked;
        }

        const hz120 = Duration(microseconds: 8333),
            hz60 = Duration(microseconds: 16667);
        expect(
          await frames(const [1, 1, 1, 1], hz120),
          inInclusiveRange(29, 32),
        );
        expect(
          await frames(const [1, 1, 1, 1], hz60),
          inInclusiveRange(29, 32),
        );
        expect(
          await frames(const [0.25, 0.25, 0.25, 0.25], hz120),
          inInclusiveRange(14, 17),
        );
      },
    );

    testWidgets('a paused session stops breathing ahead after one second', (
      tester,
    ) async {
      await tester.pumpWidget(
        host(
          WorldScene(
            background: MemoryImage(_png),
            aspect: 9 / 16,
            timeline: const WorldTimeline(
              breath: [0, 0, 0, 0],
              visibility: [1, 1, 0.5, 0],
            ),
            positionSeconds: () => 0,
          ),
        ),
      );
      await tester.pump(const Duration(seconds: 3));
      expect(
        tester.widget<Opacity>(find.byType(Opacity)).opacity,
        closeTo(1.0, 0.01),
      );
    });

    List<Matrix4> transforms(WidgetTester tester) => [
      for (final t in tester.widgetList<Transform>(find.byType(Transform)))
        t.transform,
    ];

    testWidgets(
      'nothing pulses: layers only move and turn, a butterfly\'s wings fold sideways',
      (tester) async {
        await tester.pumpWidget(
          host(
            WorldScene(
              background: MemoryImage(_png),
              aspect: 9 / 16,
              layers: [
                WorldLayer(
                  image: MemoryImage(_png),
                  depth: 'near',
                  motion: 'float',
                  attached: 'ground',
                ),
                WorldLayer(
                  image: MemoryImage(_png),
                  motion: 'sway',
                  attached: 'edge',
                  anchor: (x: 0.0, y: 0.5),
                ),
                WorldLayer(
                  image: MemoryImage(_png),
                  motion: 'flutter',
                  attached: 'air',
                  anchor: (x: 0.5, y: 0.5),
                ),
              ],
              timeline: const WorldTimeline(
                breath: [0, 1, 0, 1],
                visibility: [1, 1, 1, 1],
              ),
              positionSeconds: () => 0,
            ),
          ),
        );
        for (var i = 0; i < 40; i++) {
          await tester.pump(const Duration(milliseconds: 230));
          final layers = transforms(tester);
          expect(layers, hasLength(3));
          final [butterfly, branch, boat] = layers; // drawn farthest first
          expect(boat.getMaxScaleOnAxis(), closeTo(1.0, 1e-9));
          expect(branch.getMaxScaleOnAxis(), closeTo(1.0, 1e-9));
          expect(
            butterfly.storage[5],
            closeTo(1.0, 1e-9),
          ); // its height never changes
          expect(
            butterfly.storage[0],
            inInclusiveRange(WorldMotion.wingsFolded - 1e-6, 1.0 + 1e-9),
          );
        }
      },
    );

    testWidgets('a world that has just appeared comes alive gently', (
      tester,
    ) async {
      expect(
        (WorldScene.wake(0), WorldScene.wake(WorldScene.wakeSeconds)),
        (0.0, 1.0),
      );
      for (var t = 0.0; t < WorldScene.wakeSeconds; t += 0.25) {
        expect(
          WorldScene.wake(t + 0.25),
          greaterThanOrEqualTo(WorldScene.wake(t)),
        );
      }
      await tester.pumpWidget(
        host(
          WorldScene(
            background: MemoryImage(_png),
            aspect: 9 / 16,
            layers: [
              WorldLayer(
                image: MemoryImage(_png),
                motion: 'float',
                attached: 'ground',
              ),
            ],
            timeline: const WorldTimeline(breath: [1, 1], visibility: [1, 1]),
            positionSeconds: () => 0,
          ),
        ),
      );
      await tester.pump(const Duration(milliseconds: 16));
      // At first the boat sits exactly where it is painted (the still painting
      // it replaces), then it rides the swell.
      final first = transforms(tester).last;
      expect(first.getTranslation().y.abs(), lessThan(0.01));
      expect(first.storage[1].abs(), lessThan(1e-4));
      var moved = 0.0;
      for (var i = 0; i < 120; i++) {
        await tester.pump(const Duration(milliseconds: 50));
        moved = math.max(
          moved,
          transforms(tester).last.getTranslation().y.abs(),
        );
      }
      expect(moved, greaterThan(844 * WorldMotion.bob * 0.4));
    });

    testWidgets("an iPhone's Reduce Motion keeps the world still too", (
      tester,
    ) async {
      // iOS reports it only as the platform's reduceMotion, never as disableAnimations.
      tester.platformDispatcher.accessibilityFeaturesTestValue =
          const FakeAccessibilityFeatures(reduceMotion: true);
      addTearDown(
        tester.platformDispatcher.clearAccessibilityFeaturesTestValue,
      );
      await tester.pumpWidget(
        host(
          WorldScene(
            background: MemoryImage(_png),
            aspect: 9 / 16,
            layers: [
              WorldLayer(
                image: MemoryImage(_png),
                motion: 'float',
                attached: 'ground',
              ),
            ],
            timeline: const WorldTimeline(breath: [1, 1], visibility: [1, 1]),
            positionSeconds: () => 0,
          ),
        ),
      );
      for (var i = 0; i < 40; i++) {
        await tester.pump(const Duration(milliseconds: 100));
        final layer = transforms(tester).last;
        expect(layer.getTranslation().y, 0);
        expect(layer.storage[1], closeTo(0, 1e-12));
      }
    });

    testWidgets('reduce motion: nothing moves at all', (tester) async {
      await tester.pumpWidget(
        host(
          WorldScene(
            background: MemoryImage(_png),
            aspect: 9 / 16,
            layers: [
              WorldLayer(
                image: MemoryImage(_png),
                depth: 'near',
                motion: 'sway',
              ),
            ],
            timeline: const WorldTimeline(breath: [1, 1], visibility: [1, 1]),
            positionSeconds: () => 0,
          ),
          reduceMotion: true,
        ),
      );
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
      expect(
        (plate.left, plate.top),
        (layerBox.left, layerBox.top),
      ); // the camera stays put
    });

    testWidgets(
      'the light comes as two coats in the same scene: no layer of its own, nothing rebuilt',
      (tester) async {
        bool coated() => find
            .byWidgetPredicate(
              (w) =>
                  w is CustomPaint &&
                  w.painter.runtimeType.toString() == '_LightCoats',
            )
            .evaluate()
            .isNotEmpty;
        var at = 0.0;
        await tester.pumpWidget(
          host(
            WorldScene(
              background: MemoryImage(_png),
              aspect: 9 / 16,
              living: const WorldLiving(trees: true),
              timeline: WorldTimeline.build(durationSeconds: 600),
              positionSeconds: () => at,
              light: LightJourney.deepen,
            ),
          ),
        );
        await tester.pump(const Duration(milliseconds: 40));
        final shader = tester.state(find.byType(LivingBackground));
        expect(coated(), isFalse); // as painted at the start
        at = 30; // the light has begun
        for (var i = 0; i < 3; i++) {
          await tester.pump(const Duration(milliseconds: 40));
        }
        expect(coated(), isTrue);
        expect(find.byType(ColorFiltered), findsNothing);
        expect(
          tester.state(find.byType(LivingBackground)),
          same(shader),
        ); // the living shader keeps running
      },
    );
  });
}
