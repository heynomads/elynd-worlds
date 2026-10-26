import 'dart:math' as math;

import 'package:flutter/foundation.dart';

/// Where one layer sits at one moment, on top of the camera's drift. Pure, so
/// it can be tested and reasoned about without a widget.
@immutable
class LayerPose {
  const LayerPose({this.rotation = 0, this.dx = 0, this.dy = 0});

  /// Radians about the layer's anchor.
  final double rotation;

  /// Offsets as fractions of the painting's height.
  final double dx;
  final double dy;

  static const still = LayerPose();
}

/// How a world moves: slowly, and only as such things move in nature.
///
/// Nothing ever grows or shrinks (a boat and a stone that grew and shrank
/// with each breath moved like a heartbeat). Still things stay still. A boat on the water rises and
/// falls a little and rolls a little. Something in the air (a butterfly, a
/// leaf) makes a slow, small loop. Plants and branches sway in a light wind,
/// about where they grow from. Mist and clouds drift sideways.
///
/// The whole world drifts like a very slow camera, near layers a little more
/// than the background: depth, which is what the cut-out layers and the clean
/// plate behind them are for.
class WorldMotion {
  /// The painting covers the screen with this much to spare, for the drift.
  static const cameraHeadroom = 1.03;

  /// The camera's drift, as a fraction of the painting's height: one slow
  /// sideways sweep every 47 s and a smaller up-and-down one every 61 s.
  static const cameraDrift = 0.008;

  /// Near layers drift this much more than the background, mid ones less.
  static double parallax(String depth) => depth == 'near' ? 1.6 : 1.25;

  /// A boat on the water: rises and falls (fraction of the height) and rolls.
  static const bob = 0.0022;
  static const bobRoll = 0.4 * math.pi / 180;

  /// Something in the air: a slow loop this wide (fraction of the height).
  static const hover = 0.003;

  /// A plant or a branch in a light wind, at most this far either way.
  static const sway = 0.7 * math.pi / 180;

  /// Mist or a cloud: this far either way (fraction of the height).
  static const drift = 0.006;

  /// The camera's offset at `clockSeconds`, as fractions of the painting's height.
  static ({double dx, double dy}) camera(double clockSeconds) => (
        dx: cameraDrift * math.sin(2 * math.pi * clockSeconds / 47),
        dy: cameraDrift * 0.5 * math.sin(2 * math.pi * clockSeconds / 61 + 1.3),
      );

  /// A layer's own motion; the scene adds the camera's drift. `attached` is
  /// how the object rests ("ground" — on land or water, "edge", "air").
  /// `index` staggers the layers so they never move in step.
  static LayerPose layer({
    required String motion,
    String attached = 'ground',
    required double clockSeconds,
    int index = 0,
  }) {
    final phase = index * 1.7;
    double wave(double period, [double offset = 0]) =>
        math.sin(2 * math.pi * clockSeconds / period + phase + offset);
    switch (motion) {
      case 'float' when attached == 'ground':
        // Two slow swells, so it never rises and falls like a metronome.
        return LayerPose(
          dy: -bob * (0.75 * wave(8.7) + 0.25 * wave(4.9, 0.6)),
          rotation: bobRoll * wave(10.9, 1.1),
        );
      case 'float':
        return LayerPose(dx: hover * wave(13.7), dy: hover * 0.8 * wave(9.1, 0.4));
      case 'sway':
        // A light wind: a slow sway with a smaller, quicker one on top.
        return LayerPose(rotation: sway * (0.65 * wave(6.8) + 0.35 * wave(3.7, 1.9)));
      case 'drift':
        return LayerPose(dx: drift * wave(37));
      default:
        return LayerPose.still;
    }
  }
}
