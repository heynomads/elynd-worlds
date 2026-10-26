import 'dart:math' as math;

import 'package:flutter/foundation.dart';

/// How one layer sits at one moment. Pure, so it can be tested and reasoned
/// about without a widget.
@immutable
class LayerPose {
  const LayerPose({this.scale = 1, this.rotation = 0, this.dx = 0, this.dy = 0});

  /// Scale about the layer's anchor.
  final double scale;

  /// Radians about the layer's anchor.
  final double rotation;

  /// Offsets as fractions of the painting's height.
  final double dx;
  final double dy;
}

/// How far things move. Small on purpose: the world should be felt, not watched.
class WorldMotion {
  /// Growth of a near layer on a full in-breath, about its anchor. Growing
  /// about the base covers the layer's own outline, so a grounded object never
  /// uncovers the plate behind it when it breathes.
  static const breathScale = 0.03;

  /// The whole world breathes a little too, like a slow camera.
  static const cameraScale = 0.008;
  static const cameraHeadroom = 1.02;

  /// Sway: a branch-like rotation about where the layer is attached.
  static const swayRadians = 0.6 * math.pi / 180;

  /// Drift and float, as fractions of the painting's height.
  static const drift = 0.003;
  static const float = 0.004;

  /// Parallax: near layers drift this much more than the background.
  static const parallax = 0.002;

  static double depthWeight(String depth) => depth == 'near' ? 1.0 : 0.6;

  /// The pose of one layer. `index` staggers the layers so they never move in step.
  static LayerPose layer({
    required String depth,
    required String motion,
    required double breath,
    required double clockSeconds,
    int index = 0,
  }) {
    final w = depthWeight(depth);
    final phase = index * 1.7;
    final scale = 1 + breathScale * w * breath;
    final rotation = motion == 'sway' ? swayRadians * w * math.sin(2 * math.pi * clockSeconds / 9 + phase) : 0.0;
    final drifting = motion == 'drift' ? drift * w * math.sin(2 * math.pi * clockSeconds / 16 + phase) : 0.0;
    final floating = motion == 'float' ? float * math.sin(2 * math.pi * clockSeconds / 7 + phase) : 0.0;
    final parallaxDx = parallax * w * math.sin(2 * math.pi * clockSeconds / 23);
    return LayerPose(scale: scale, rotation: rotation, dx: drifting + parallaxDx, dy: -floating);
  }

  /// The whole world's slow breathing, about its centre.
  static double camera(double breath) => cameraHeadroom + cameraScale * breath;
}
