import 'dart:math' as math;

import 'package:flutter/foundation.dart';

/// Where one layer sits at one moment. Pure, so it can be tested and reasoned
/// about without a widget.
@immutable
class LayerPose {
  const LayerPose({
    this.rotation = 0,
    this.dx = 0,
    this.dy = 0,
    this.wings = 1,
  });

  /// Radians about the layer's anchor.
  final double rotation;

  /// Offsets as fractions of the painting's height.
  final double dx;
  final double dy;

  /// A fluttering creature's wings, as a share of their open width (1 = open):
  /// they fold now and then, sideways about the body. Nothing else ever
  /// changes size.
  final double wings;

  static const still = LayerPose();

  /// This pose `k` of the way from still (0) to full (1).
  LayerPose scaled(double k) => LayerPose(
    rotation: rotation * k,
    dx: dx * k,
    dy: dy * k,
    wings: 1 - (1 - wings) * k,
  );
}

/// How a world's objects move: as such things move in nature, clearly enough to
/// see: a boat that grew and shrank with each breath moved like a heartbeat,
/// and a world that barely moved looked like a static picture.
///
/// A boat on the water rides the ripples: it sinks a little and rises back to
/// where it is painted, and rolls a little. Never higher than painted, so
/// whatever the cut left on the painting around it (an oar, the end of a rope)
/// meets it again at the top of every swell. Something in the air floats on a slow loop. A butterfly or a moth
/// also folds its wings now and then. Plants and branches sway in gusts of
/// wind. Mist drifts. Stones, posts, cabins and lanterns stay still (a lantern
/// flickers instead, see `living.frag`). The water, the sky, the grass and the
/// trees around them move in the shader. The values follow the reference
/// renderer (the first prototype), as fractions of the
/// painting's height.
/// How long each motion takes to come round. A meditation's world uses [live]: periods that share no common
/// cycle, so it never repeats itself (a product decision, 29 Oct 2026: the world during a session must not loop). Only the
/// shared clip's renderer uses [clip], where every period divides 12 s, so its 12 seconds loop without a seam.
class WorldPeriods {
  const WorldPeriods({
    required this.swell,
    required this.swellShort,
    required this.roll,
    required this.hoverX,
    required this.hoverY,
    required this.wing,
    required this.sway,
    required this.swayShort,
    required this.drift,
  });

  final double swell, swellShort, roll, hoverX, hoverY, wing, sway, swayShort, drift;

  static const live = WorldPeriods(
      swell: 4.3, swellShort: 2.7, roll: 5.1, hoverX: 7.9, hoverY: 5.3, wing: 4.2, sway: 4.6, swayShort: 1.9, drift: 14);
  static const clip = WorldPeriods(
      swell: 4.0, swellShort: 3.0, roll: 6.0, hoverX: 6.0, hoverY: 4.0, wing: 4.0, sway: 4.0, swayShort: 2.0, drift: 12);
}

class WorldMotion {
  /// The periods in use: [WorldPeriods.live] everywhere; the clip renderer (test/render_preview_test.dart with
  /// LOOP=true, run by `elynd-worlds loop`) sets [WorldPeriods.clip].
  static WorldPeriods periods = WorldPeriods.live;

  /// A boat on the water: sinks this far at most (two swells), and rolls.
  static const bob = 0.006;
  static const bobRoll = 0.8 * math.pi / 180;

  /// Something in the air: a slow loop this wide and this high.
  static const hoverX = 0.010;
  static const hoverY = 0.007;

  /// A butterfly folds its wings to this share of their width, for a moment.
  static const wingsFolded = 0.45;
  static double get wingCycle => periods.wing;
  static const wingFold = 0.55;

  /// A plant or a branch in gusts of wind, at most this far either way.
  static const sway = 2.0 * math.pi / 180;

  /// Mist or a cloud: this far either way.
  static const drift = 0.010;

  /// The shared clip's length: every [WorldPeriods.clip] period divides it, so the clip loops without a seam.
  static const loopSeconds = 12.0;

  /// A layer's motion at `clockSeconds`. `attached` is how the object rests
  /// ("ground" — on land or water, "edge", "air"); `index` staggers the layers
  /// so they never move in step.
  static LayerPose layer({
    required String motion,
    String attached = 'ground',
    required double clockSeconds,
    int index = 0,
  }) {
    final t = clockSeconds;
    final p = periods;
    final phase = index * 1.7;
    double wave(double period, [double offset = 0]) =>
        math.sin(2 * math.pi * t / period + phase + offset);
    switch (motion) {
      case 'float' when attached == 'ground':
        final swell = 0.7 * wave(p.swell) + 0.3 * math.sin(2 * math.pi * t / p.swellShort);
        return LayerPose(
          dy: bob * (0.5 + 0.5 * swell),
          rotation: bobRoll * wave(p.roll, 0.8),
        );
      case 'float':
        return LayerPose(dx: hoverX * wave(p.hoverX), dy: hoverY * wave(p.hoverY));
      case 'flutter':
        final cycle = (t + phase) % wingCycle;
        final fold = cycle < wingFold
            ? math.pow(math.sin(math.pi * cycle / wingFold), 2).toDouble()
            : 0.0;
        return LayerPose(
          dx: hoverX * wave(p.hoverX),
          dy: hoverY * wave(p.hoverY),
          wings: 1 - (1 - wingsFolded) * fold,
        );
      case 'sway':
        return LayerPose(
          rotation: sway * (0.65 * wave(p.sway) + 0.35 * wave(p.swayShort)),
        );
      case 'drift':
        return LayerPose(dx: drift * wave(p.drift));
      default:
        return LayerPose.still;
    }
  }
}
