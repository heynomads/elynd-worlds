import 'dart:async';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/widgets.dart';

/// A light that flickers (a lantern, a candle), as fractions of the painting.
@immutable
class WorldLight {
  const WorldLight({required this.x, required this.y, required this.size});

  static WorldLight? fromJson(Object? json) {
    if (json is! Map) return null;
    final x = json['x'], y = json['y'], size = json['size'];
    if (x is! num || y is! num || size is! num) return null;
    return WorldLight(
      x: x.toDouble().clamp(0.0, 1.0),
      y: y.toDouble().clamp(0.0, 1.0),
      size: size.toDouble().clamp(0.0, 1.0),
    );
  }

  final double x;
  final double y;
  final double size;
}

/// What moves in nature in one world (the manifest's "living" block), ready to draw.
///
/// [regions] is the region map SAM 3.1 made: red = water, green = sky,
/// blue = grass, alpha = stream. [trees] sways the upper part of the painting
/// outside the sky. [lights] flicker. [particles] drift through the air.
@immutable
class WorldLiving {
  const WorldLiving({
    this.regions,
    this.water = false,
    this.sky = false,
    this.grass = false,
    this.stream = false,
    this.trees = false,
    this.horizon,
    this.lights = const [],
    this.particles = 'none',
  });

  final ImageProvider? regions;
  final bool water;
  final bool sky;
  final bool grass;
  final bool stream;
  final bool trees;

  /// Where the water starts, as a fraction of the painting's height.
  final double? horizon;
  final List<WorldLight> lights;

  /// "leaves" | "snow" | "pollen" | "fireflies" | "none".
  final String particles;

  bool get mapped => regions != null && (water || sky || grass || stream);

  /// Anything for the shader to do (a map, swaying crowns or a light).
  bool get shaded => mapped || trees || lights.isNotEmpty;
}

/// The plate, moving as nature moves (shaders/living.frag). Falls back to the
/// plate as painted while the shader or the pictures are not ready, or if the
/// shader cannot load on this device.
class LivingBackground extends StatefulWidget {
  const LivingBackground({
    super.key,
    required this.plate,
    required this.living,
    required this.clockSeconds,
    this.still = false,
  });

  final ImageProvider plate;
  final WorldLiving living;
  final double clockSeconds;

  /// Reduce motion: the painting as painted.
  final bool still;

  /// The shader, loaded once per app run. Null when it cannot load.
  static Future<ui.FragmentProgram?> get program =>
      _program ??= ui.FragmentProgram.fromAsset(
        shaderAsset,
      ).then<ui.FragmentProgram?>((p) => p).catchError((Object _) => null);
  static Future<ui.FragmentProgram?>? _program;

  /// The asset key of the shader (declared under flutter: shaders: in the pubspec).
  static const shaderAsset = 'shaders/living.frag';

  @visibleForTesting
  static void debugReset() => _program = null;

  @override
  State<LivingBackground> createState() => _LivingBackgroundState();
}

class _LivingBackgroundState extends State<LivingBackground> {
  ui.FragmentProgram? _program;
  ui.Image? _plate;
  ui.Image? _regions;
  ImageStream? _plateStream;
  ImageStream? _regionsStream;
  ImageStreamListener? _plateListener;
  ImageStreamListener? _regionsListener;

  @override
  void initState() {
    super.initState();
    LivingBackground.program.then((p) {
      if (mounted) setState(() => _program = p);
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _resolve();
  }

  @override
  void didUpdateWidget(LivingBackground old) {
    super.didUpdateWidget(old);
    if (old.plate != widget.plate ||
        old.living.regions != widget.living.regions) {
      _resolve();
    }
  }

  void _resolve() {
    final config = createLocalImageConfiguration(context);
    _plateStream = _listen(
      widget.plate,
      config,
      _plateStream,
      _plateListener,
      (image) => _plate = image,
      (l) => _plateListener = l,
    );
    final regions = widget.living.regions;
    if (regions != null) {
      _regionsStream = _listen(
        regions,
        config,
        _regionsStream,
        _regionsListener,
        (image) => _regions = image,
        (l) => _regionsListener = l,
      );
    }
  }

  ImageStream _listen(
    ImageProvider provider,
    ImageConfiguration config,
    ImageStream? old,
    ImageStreamListener? oldListener,
    void Function(ui.Image) keep,
    void Function(ImageStreamListener) remember,
  ) {
    final stream = provider.resolve(config);
    if (old?.key == stream.key && oldListener != null) return old!;
    if (old != null && oldListener != null) old.removeListener(oldListener);
    final listener = ImageStreamListener((info, _) {
      if (mounted) setState(() => keep(info.image));
    });
    remember(listener);
    stream.addListener(listener);
    return stream;
  }

  @override
  void dispose() {
    if (_plateListener != null) _plateStream?.removeListener(_plateListener!);
    if (_regionsListener != null) {
      _regionsStream?.removeListener(_regionsListener!);
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final program = _program, plate = _plate;
    final living = widget.living;
    final regions = living.mapped ? _regions : null;
    final ready =
        program != null && plate != null && (!living.mapped || regions != null);
    if (!ready) {
      return Image(
        image: widget.plate,
        fit: BoxFit.fill,
        gaplessPlayback: true,
      );
    }
    return CustomPaint(
      painter: _LivingPainter(
        program: program,
        plate: plate,
        regions:
            regions ??
            plate, // unmapped: a placeholder the shader never reads (all regions off)
        living: living,
        mapped: regions != null,
        // Wrapped to an hour: the shader's sines stay exact on any GPU.
        time: widget.clockSeconds % 3600,
        motion: widget.still ? 0 : 1,
      ),
      size: Size.infinite,
    );
  }
}

class _LivingPainter extends CustomPainter {
  _LivingPainter({
    required this.program,
    required this.plate,
    required this.regions,
    required this.living,
    required this.mapped,
    required this.time,
    required this.motion,
  });

  final ui.FragmentProgram program;
  final ui.Image plate;
  final ui.Image regions;
  final WorldLiving living;
  final bool mapped;
  final double time;
  final double motion;

  @override
  void paint(Canvas canvas, Size size) {
    final shader = program.fragmentShader();
    final light = living.lights.isEmpty ? null : living.lights.first;
    double on(bool v) => v && mapped ? 1 : 0;
    final values = <double>[
      size.width, size.height, // uSize
      0, 0, // uOrigin: the canvas is the painting's rect
      time, motion,
      on(living.water),
      on(living.sky),
      on(living.grass),
      on(living.stream), // uOn
      living.trees ? 1 : 0, living.horizon ?? -1, 0, 0, // uScene
      light?.x ?? 0,
      light?.y ?? 0,
      light?.size ?? 0,
      light == null ? 0 : 1, // uLight
    ];
    for (var i = 0; i < values.length; i++) {
      shader.setFloat(i, values[i]);
    }
    shader
      ..setImageSampler(0, plate)
      ..setImageSampler(1, regions);
    canvas.drawRect(Offset.zero & size, Paint()..shader = shader);
  }

  @override
  bool shouldRepaint(_LivingPainter old) =>
      old.time != time ||
      old.motion != motion ||
      old.plate != plate ||
      old.regions != regions ||
      old.living != living;
}

/// What drifts through the air: leaves (autumn), snow, pollen (a summer meadow) or
/// fireflies (a warm night). A few, slow, each on its own path; the same at the
/// same time, so a world looks the same on every phone.
class WorldParticles extends CustomPainter {
  WorldParticles({
    required this.kind,
    required this.clockSeconds,
    this.palette = const [],
  });

  final String kind;
  final double clockSeconds;
  final List<Color> palette;

  static const _count = {
    'leaves': 7,
    'snow': 28,
    'pollen': 18,
    'fireflies': 12,
  };

  @override
  void paint(Canvas canvas, Size size) {
    final n = _count[kind] ?? 0;
    for (var i = 0; i < n; i++) {
      final r = math.Random(7 + i * 31);
      final x0 = r.nextDouble(),
          start = r.nextDouble() * 40,
          s = r.nextDouble();
      switch (kind) {
        case 'leaves':
          _leaf(canvas, size, i, x0, start, s);
        case 'snow':
          _snow(canvas, size, x0, start, s);
        case 'pollen':
          _mote(canvas, size, x0, s, i, const Color(0xFFFFF4D6), 1.4, 0.55);
        case 'fireflies':
          _firefly(canvas, size, x0, s, i);
      }
    }
  }

  void _leaf(
    Canvas canvas,
    Size size,
    int i,
    double x0,
    double start,
    double s,
  ) {
    final cycle =
        size.height * 0.85 / (26 + 16 * s) +
        2; // seconds to fall through the picture
    final age = (clockSeconds + start) % cycle;
    final y = -20 + age * (26 + 16 * s) * size.height / 960;
    final x =
        x0 * size.width + 18 * size.width / 540 * math.sin(age * 1.3 + start);
    final colors = palette.isNotEmpty
        ? palette
        : const [Color(0xFFD69630), Color(0xFFE2AC40), Color(0xFFBE6E28)];
    final w = (5 + 3 * s) * size.width / 540;
    final tilt = math.sin(age * (1.5 + 1.5 * s)).abs();
    canvas.save();
    canvas.translate(x, y);
    canvas.rotate(math.sin(age * 0.9 + i) * 0.8);
    canvas.drawOval(
      Rect.fromCenter(
        center: Offset.zero,
        width: 2 * w,
        height: w * (0.35 + 0.65 * tilt),
      ),
      Paint()..color = colors[i % colors.length].withValues(alpha: 0.82),
    );
    canvas.restore();
  }

  void _snow(Canvas canvas, Size size, double x0, double start, double s) {
    final speed = (18 + 22 * s) * size.height / 960;
    final cycle = size.height / speed + 1;
    final age = (clockSeconds + start) % cycle;
    final y = age * speed - 6;
    final x = x0 * size.width + 10 * math.sin(age * 0.8 + start);
    canvas.drawCircle(
      Offset(x, y),
      (1.2 + 1.8 * s) * size.width / 540,
      Paint()..color = const Color(0xFFFFFFFF).withValues(alpha: 0.75),
    );
  }

  void _mote(
    Canvas canvas,
    Size size,
    double x0,
    double s,
    int i,
    Color color,
    double radius,
    double alpha,
  ) {
    final t = clockSeconds;
    final x = (x0 + 0.04 * math.sin(t / (5 + 3 * s) + i)) * size.width;
    final y =
        (0.35 + 0.45 * s + 0.03 * math.sin(t / (4 + 2 * s) + i * 1.3)) *
        size.height;
    canvas.drawCircle(
      Offset(x, y),
      radius * size.width / 540,
      Paint()
        ..color = color.withValues(
          alpha: alpha * (0.6 + 0.4 * math.sin(t * 0.7 + i)),
        ),
    );
  }

  void _firefly(Canvas canvas, Size size, double x0, double s, int i) {
    final t = clockSeconds;
    final x = (x0 + 0.05 * math.sin(t / (3 + 2 * s) + i)) * size.width;
    final y =
        (0.45 + 0.4 * s + 0.04 * math.sin(t / (2.5 + s) + i * 0.7)) *
        size.height;
    final glow = math.max(0.0, math.sin(t * (0.8 + 0.6 * s) + i * 2.1));
    if (glow <= 0) return;
    final r = 2.2 * size.width / 540;
    canvas.drawCircle(
      Offset(x, y),
      r * 3,
      Paint()
        ..color = const Color(0xFFFFE27A).withValues(alpha: 0.25 * glow)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 4),
    );
    canvas.drawCircle(
      Offset(x, y),
      r,
      Paint()..color = const Color(0xFFFFF3B0).withValues(alpha: 0.9 * glow),
    );
  }

  @override
  bool shouldRepaint(WorldParticles old) =>
      old.clockSeconds != clockSeconds || old.kind != kind;
}
