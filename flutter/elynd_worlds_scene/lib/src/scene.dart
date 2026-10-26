import 'package:flutter/scheduler.dart';
import 'package:flutter/widgets.dart';

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
    this.anchor = (x: 0.5, y: 1.0),
  });

  final ImageProvider image;
  final String depth;
  final String motion;

  /// The pivot of the layer's motion, as fractions of the painting.
  final ({double x, double y}) anchor;
}

/// A painted world that breathes with a meditation.
///
/// Draws `background` (the plate: the painting without its subjects, or the
/// full painting for a still world) and the layers on top, all covering the
/// widget like `BoxFit.cover`. Each frame reads the session's position from
/// `positionSeconds` (a callback, so the host never notifies anything while it
/// builds) and poses every layer about its own anchor. Respects the platform's
/// reduce-motion setting: the world then stays still and only fades.
class WorldScene extends StatefulWidget {
  const WorldScene({
    super.key,
    required this.background,
    required this.aspect,
    required this.timeline,
    required this.positionSeconds,
    this.layers = const [],
    this.followVisibility = true,
  });

  final ImageProvider background;

  /// Width / height of the painting.
  final double aspect;
  final List<WorldLayer> layers;
  final WorldTimeline timeline;
  final double Function() positionSeconds;

  /// Fade with the timeline's visibility (eyes closed → a glow).
  final bool followVisibility;

  @override
  State<WorldScene> createState() => _WorldSceneState();
}

class _WorldSceneState extends State<WorldScene> with SingleTickerProviderStateMixin {
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
  /// at most one second ahead (a paused session must not keep breathing on).
  double get _seconds {
    final since = (_clock - _positionChangedAt).inMicroseconds / 1e6;
    return _position + since.clamp(0.0, 1.0);
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  static Rect _cover(Size box, double aspect) {
    final width = box.width / box.height > aspect ? box.width : box.height * aspect;
    final height = width / aspect;
    return Rect.fromCenter(center: box.center(Offset.zero), width: width, height: height);
  }

  @override
  Widget build(BuildContext context) {
    final still = MediaQuery.maybeDisableAnimationsOf(context) ?? false;
    final t = _seconds;
    final breath = still ? 0.0 : widget.timeline.breathAt(t);
    final clock = still ? 0.0 : _clock.inMicroseconds / 1e6;
    final visibility = widget.followVisibility ? widget.timeline.visibilityAt(t).clamp(0.0, 1.0) : 1.0;

    return LayoutBuilder(builder: (context, box) {
      final rect = _cover(box.biggest, widget.aspect);
      final children = <Widget>[
        Positioned.fromRect(rect: rect, child: Image(image: widget.background, fit: BoxFit.fill, gaplessPlayback: true)),
        for (var i = widget.layers.length - 1; i >= 0; i--) // farthest first, nearest on top
          Positioned.fromRect(rect: rect, child: _layer(widget.layers[i], i, breath, clock, rect.height)),
      ];
      return ClipRect(
        child: Opacity(
          opacity: visibility,
          child: Transform.scale(
            scale: still ? WorldMotion.cameraHeadroom : WorldMotion.camera(breath),
            child: Stack(clipBehavior: Clip.none, children: children),
          ),
        ),
      );
    });
  }

  Widget _layer(WorldLayer layer, int index, double breath, double clock, double height) {
    final pose = WorldMotion.layer(depth: layer.depth, motion: layer.motion, breath: breath, clockSeconds: clock, index: index);
    final matrix = Matrix4.identity()
      ..translateByDouble(pose.dx * height, pose.dy * height, 0, 1)
      ..rotateZ(pose.rotation)
      ..scaleByDouble(pose.scale, pose.scale, 1, 1);
    return Transform(
      alignment: Alignment(layer.anchor.x * 2 - 1, layer.anchor.y * 2 - 1),
      transform: matrix,
      child: Image(image: layer.image, fit: BoxFit.fill, gaplessPlayback: true),
    );
  }
}
