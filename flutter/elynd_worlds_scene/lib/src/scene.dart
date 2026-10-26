import 'dart:math' as math;

import 'package:flutter/scheduler.dart';
import 'package:flutter/widgets.dart';

import 'living.dart';
import 'motion.dart';
import 'timeline.dart';

/// One cut-out layer ready to draw: the image (the painting's size, transparent
/// outside the subject) and how it moves.
@immutable
class WorldLayer {
  const WorldLayer({
    required this.image,
    this.depth = 'mid',
    this.motion = 'still',
    this.attached = 'ground',
    this.glows = false,
    this.anchor = (x: 0.5, y: 1.0),
  });

  final ImageProvider image;
  final String depth;
  final String motion;

  /// How the object rests: "ground" (on land or on water), "edge" or "air".
  /// A floating boat bobs; a floating butterfly loops.
  final String attached;

  /// A light that glows by itself (a lantern, a candle): it flickers.
  final bool glows;

  /// The pivot of the layer's motion, as fractions of the painting.
  final ({double x, double y}) anchor;
}

/// A painted world that lives, the way nature does, with a meditation.
///
/// Draws `background` (the plate: the painting without its subjects, or the
/// full painting for a still world) and the layers on top, all covering the
/// widget like `BoxFit.cover`. With [living], the parts of the painting that
/// move in nature move (the water ripples, the clouds drift, the grass waves,
/// a stream flows, the crowns sway, a lantern flickers: `living.frag`), and
/// leaves, snow, pollen or fireflies drift through the air. Each layer moves
/// as its object would ([WorldMotion]); nothing pulses. A world that has just
/// appeared comes alive over [wakeSeconds]. Each frame reads the
/// session's position from `positionSeconds` (a callback, so the host never
/// notifies anything while it builds) for the fading: the whole world while
/// the listener arrives, a glow while their eyes are closed. Respects the
/// platform's reduce-motion setting: the world then stays still and only fades.
class WorldScene extends StatefulWidget {
  const WorldScene({
    super.key,
    required this.background,
    required this.aspect,
    required this.timeline,
    required this.positionSeconds,
    this.layers = const [],
    this.living,
    this.followVisibility = true,
    this.focus = (x: 0.5, y: 0.5),
  });

  final ImageProvider background;

  /// Width / height of the painting.
  final double aspect;
  final List<WorldLayer> layers;

  /// What moves in nature; null: only the layers move.
  final WorldLiving? living;
  final WorldTimeline timeline;
  final double Function() positionSeconds;

  /// Fade with the timeline's visibility (eyes closed → a glow).
  final bool followVisibility;

  /// Where the subjects are (fractions of the painting). A screen narrower or
  /// shorter than the painting crops around this point, not around the middle.
  final ({double x, double y}) focus;

  /// The painting's rect covering `box`, its overflow placed so that `focus`
  /// sits as near the middle as the edges allow. A focus lower than `highest`
  /// (subjects painted low, behind a player's words and controls) is lifted to
  /// it by zooming in a little, at most `maxZoom`.
  static Rect cover(
    Size box,
    double aspect,
    ({double x, double y}) focus, {
    double highest = 0.62,
    double maxZoom = 1.25,
  }) {
    var width = box.width / box.height > aspect
        ? box.width
        : box.height * aspect;
    var height = width / aspect;
    if (focus.y > highest && focus.y < 1) {
      final zoom = (box.height * (1 - highest) / (height * (1 - focus.y)))
          .clamp(1.0, maxZoom);
      width *= zoom;
      height *= zoom;
    }
    final targetY = box.height * (focus.y > highest ? highest : focus.y);
    final left = (box.width / 2 - focus.x * width).clamp(
      box.width - width,
      0.0,
    );
    final top = (targetY - focus.y * height).clamp(box.height - height, 0.0);
    return Rect.fromLTWH(left, top, width, height);
  }

  /// A world that has just appeared comes alive over this many seconds, from
  /// the painting as painted to its full motion: it never starts mid-swell.
  static const wakeSeconds = 3.0;

  /// How alive the world is `clockSeconds` after it appeared (0 to 1, eased).
  static double wake(double clockSeconds) =>
      Curves.easeInOut.transform((clockSeconds / wakeSeconds).clamp(0.0, 1.0));

  /// A lantern's flicker (the same as the light it throws in `living.frag`).
  static double flicker(double t) =>
      math.sin(2 * math.pi * 1.7 * t) * 0.5 +
      math.sin(2 * math.pi * 2.9 * t + 1.1) * 0.3 +
      math.sin(2 * math.pi * 5.3 * t + 2.3) * 0.2;

  @override
  State<WorldScene> createState() => _WorldSceneState();
}

class _WorldSceneState extends State<WorldScene>
    with SingleTickerProviderStateMixin {
  late final Ticker _ticker;
  Duration _clock = Duration.zero;
  double _position = 0;
  Duration _positionChangedAt = Duration.zero;

  @override
  void initState() {
    super.initState();
    _position = widget.positionSeconds();
    _ticker = createTicker(_tick)..start();
  }

  void _tick(Duration elapsed) {
    final position = widget.positionSeconds();
    if (position != _position) {
      _position = position;
      _positionChangedAt = elapsed;
    }
    setState(() => _clock = elapsed);
  }

  /// Session seconds now: the last position plus the time since it changed,
  /// at most one second ahead (a paused session must not keep fading on).
  double get _seconds {
    final since = (_clock - _positionChangedAt).inMicroseconds / 1e6;
    return _position + since.clamp(0.0, 1.0);
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final still = MediaQuery.maybeDisableAnimationsOf(context) ?? false;
    final clock = still ? 0.0 : _clock.inMicroseconds / 1e6;
    final wake = still ? 0.0 : WorldScene.wake(clock);
    final visibility = widget.followVisibility
        ? widget.timeline.visibilityAt(_seconds).clamp(0.0, 1.0)
        : 1.0;
    final living = widget.living;

    return LayoutBuilder(
      builder: (context, box) {
        final rect = WorldScene.cover(box.biggest, widget.aspect, widget.focus);
        final h = rect.height;
        final children = <Widget>[
          Positioned.fromRect(
            rect: rect,
            child: living != null && living.shaded
                ? LivingBackground(
                    plate: widget.background,
                    living: living,
                    clockSeconds: clock,
                    still: still,
                    motion: wake,
                  )
                : Image(
                    image: widget.background,
                    fit: BoxFit.fill,
                    gaplessPlayback: true,
                  ),
          ),
          for (
            var i = widget.layers.length - 1;
            i >= 0;
            i--
          ) // farthest first, nearest on top
            Positioned.fromRect(
              rect: rect,
              child: _layer(widget.layers[i], i, still, clock, wake, h),
            ),
          if (living != null && living.particles != 'none' && !still)
            Positioned.fromRect(
              rect: rect,
              child: IgnorePointer(
                child: Opacity(
                  opacity: wake,
                  child: CustomPaint(
                    painter: WorldParticles(
                      kind: living.particles,
                      clockSeconds: clock,
                    ),
                  ),
                ),
              ),
            ),
        ];
        return ClipRect(
          child: Opacity(
            opacity: visibility,
            child: Stack(clipBehavior: Clip.none, children: children),
          ),
        );
      },
    );
  }

  Widget _layer(
    WorldLayer layer,
    int index,
    bool still,
    double clock,
    double wake,
    double height,
  ) {
    final pose = still
        ? LayerPose.still
        : WorldMotion.layer(
            motion: layer.motion,
            attached: layer.attached,
            clockSeconds: clock,
            index: index,
          ).scaled(wake);
    final matrix = Matrix4.identity()
      ..translateByDouble(pose.dx * height, pose.dy * height, 0, 1)
      ..rotateZ(pose.rotation)
      ..scaleByDouble(pose.wings, 1, 1, 1);
    Widget image = Image(
      image: layer.image,
      fit: BoxFit.fill,
      gaplessPlayback: true,
    );
    if (layer.glows && !still) {
      final f = 1 + 0.12 * wake * WorldScene.flicker(clock);
      image = ColorFiltered(
        colorFilter: ColorFilter.matrix(<double>[
          f,
          0,
          0,
          0,
          0,
          0,
          f,
          0,
          0,
          0,
          0,
          0,
          f,
          0,
          0,
          0,
          0,
          0,
          1,
          0,
        ]),
        child: image,
      );
    }
    return Transform(
      alignment: Alignment(layer.anchor.x * 2 - 1, layer.anchor.y * 2 - 1),
      transform: matrix,
      child: image,
    );
  }
}
