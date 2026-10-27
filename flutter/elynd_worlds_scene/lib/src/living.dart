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
/// blue = grass, alpha = 1 - stream. [trees] sways the upper part of the painting
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
    this.birds = 'none',
    this.night = false,
    this.fish = false,
    this.shadows = false,
    this.seed = 0,
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

  /// "leaves" | "snow" | "pollen" | "fireflies" | "rain" | "petals" | "none".
  final String particles;

  /// What lives there (LivingSpec): birds crossing the sky, a night sky, fish rings, cloud shadows.
  final String birds;
  final bool night;
  final bool fish;
  final bool shadows;
  final double seed;

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
    this.motion = 1,
  });

  final ImageProvider plate;
  final WorldLiving living;
  final double clockSeconds;

  /// Reduce motion: the painting as painted.
  final bool still;

  /// How much it moves: 0 the painting as painted, 1 fully alive.
  final double motion;

  /// The shader, loaded once per app run. Null when it cannot load.
  static Future<ui.FragmentProgram?> get program =>
      _program ??= ui.FragmentProgram.fromAsset(
        shaderAsset,
      ).then<ui.FragmentProgram?>((p) => p).catchError((Object _) => null);
  static Future<ui.FragmentProgram?>? _program;

  /// The asset key of the shader (declared under flutter: shaders: in the pubspec).
  static const shaderAsset = 'shaders/living.frag';

  /// The shader's clock wraps every four hours: its sines stay exact on any
  /// GPU, and no scene lasts that long, so the wrap's jump is never seen.
  static const wrapSeconds = 4 * 3600.0;

  /// A region map with nothing in it (no water, sky, grass or stream), for a
  /// world without one: its crowns and lights still move, as in a browser.
  static final ui.Image noRegions = () {
    final recorder = ui.PictureRecorder();
    Canvas(recorder).drawRect(
      const Rect.fromLTWH(0, 0, 1, 1),
      Paint()..color = const Color(0xFF000000),
    );
    final picture = recorder.endRecording();
    final image = picture.toImageSync(1, 1);
    picture.dispose();
    return image;
  }();

  @visibleForTesting
  static void debugReset() => _program = null;

  @override
  State<LivingBackground> createState() => _LivingBackgroundState();
}

class _LivingBackgroundState extends State<LivingBackground> {
  ui.FragmentProgram? _program;
  ui.FragmentShader? _shader;
  // The pictures as their streams hand them over: ours to dispose.
  ImageInfo? _plate;
  ImageInfo? _regions;
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
    _plateStream = _listen(widget.plate, config, _plateStream, _plateListener, (
      info,
    ) {
      _plate?.dispose();
      _plate = info;
    }, (l) => _plateListener = l);
    final regions = widget.living.regions;
    if (regions != null) {
      _regionsStream = _listen(
        regions,
        config,
        _regionsStream,
        _regionsListener,
        (info) {
          _regions?.dispose();
          _regions = info;
        },
        (l) => _regionsListener = l,
      );
    }
  }

  ImageStream _listen(
    ImageProvider provider,
    ImageConfiguration config,
    ImageStream? old,
    ImageStreamListener? oldListener,
    void Function(ImageInfo) keep,
    void Function(ImageStreamListener) remember,
  ) {
    final stream = provider.resolve(config);
    if (old?.key == stream.key && oldListener != null) return old!;
    if (old != null && oldListener != null) old.removeListener(oldListener);
    final listener = ImageStreamListener((info, _) {
      if (mounted) {
        setState(() => keep(info));
      } else {
        info.dispose();
      }
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
    _plate?.dispose();
    _regions?.dispose();
    _shader?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final program = _program, plate = _plate?.image;
    final living = widget.living;
    final regions = living.mapped ? _regions?.image : null;
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
        // One shader for the life of this widget, its values set each frame.
        shader: _shader ??= program.fragmentShader(),
        plate: plate,
        regions: regions ?? LivingBackground.noRegions,
        living: living,
        mapped: regions != null,
        time: widget.clockSeconds % LivingBackground.wrapSeconds,
        motion: widget.still ? 0 : widget.motion.clamp(0.0, 1.0),
      ),
      size: Size.infinite,
    );
  }
}

class _LivingPainter extends CustomPainter {
  _LivingPainter({
    required this.shader,
    required this.plate,
    required this.regions,
    required this.living,
    required this.mapped,
    required this.time,
    required this.motion,
  });

  final ui.FragmentShader shader;
  final ui.Image plate;
  final ui.Image regions;
  final WorldLiving living;
  final bool mapped;
  final double time;
  final double motion;

  @override
  void paint(Canvas canvas, Size size) {
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
      living.trees ? 1 : 0, living.horizon ?? -1,
      on(living.water) * (living.particles == 'rain' ? 1 : 0),
      living.seed, // uScene
      light?.x ?? 0,
      light?.y ?? 0,
      light?.size ?? 0,
      light == null ? 0 : 1, // uLight
      on(living.sky) * (const {'flock': 1.0, 'gulls': 2.0}[living.birds] ?? 0),
      on(living.sky) * (living.night ? 1 : 0),
      on(living.water) * (living.fish ? 1 : 0),
      on(living.grass) * (living.shadows ? 1 : 0), // uLife
    ];
    for (var i = 0; i < values.length; i++) {
      shader.setFloat(i, values[i]);
    }
    // Smooth sampling, as the Image widget draws the plate (medium): the
    // sampler's default (none, nearest) would show the painting's pixels as
    // blocks on a phone, where it is scaled up, and make moving water shimmer.
    shader
      ..setImageSampler(0, plate, filterQuality: ui.FilterQuality.medium)
      ..setImageSampler(1, regions, filterQuality: ui.FilterQuality.low);
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
    'rain': 72,
    'petals': 10,
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
        case 'rain':
          _rain(canvas, size, x0, start, s);
        case 'petals':
          _petal(canvas, size, i, x0, start, s);
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
    // From just above the picture until just past its bottom, then again.
    final speed = (26 + 16 * s) * size.height / 960;
    final age = (clockSeconds + start) % ((size.height + 40) / speed);
    final y = -20 + age * speed;
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

  /// Rain: fine streaks (a raindrop seen through a camera is a short line), slanted a little by the wind,
  /// falling fast and faint, each at its own pace.
  void _rain(Canvas canvas, Size size, double x0, double start, double s) {
    final speed = (0.9 + 0.5 * s) * size.height; // px a second
    final cycle = (size.height + 80) / speed;
    final age = (clockSeconds + start) % cycle;
    final y = -40 + age * speed;
    final x = (x0 * 1.12 - 0.06) * size.width + 0.1 * y;
    final len = (13 + 11 * s) * size.width / 540;
    canvas.drawLine(
      Offset(x, y),
      Offset(x - 0.1 * len, y - len),
      Paint()
        ..color = const Color(0xFFE6ECF2).withValues(alpha: 0.13 + 0.13 * s)
        ..strokeWidth = (0.6 + 0.5 * s) * size.width / 540
        ..strokeCap = StrokeCap.round,
    );
  }

  /// Blossom petals: small, pale pink, drifting down slowly and turning over as they fall.
  void _petal(
    Canvas canvas,
    Size size,
    int i,
    double x0,
    double start,
    double s,
  ) {
    final speed = (15 + 11 * s) * size.height / 960;
    final age = (clockSeconds + start) % ((size.height + 30) / speed);
    final y = -15 + age * speed;
    final x =
        x0 * size.width +
        size.width / 540 * (24 * math.sin(age * 0.8 + start) + 3.5 * age);
    final w = (3.0 + 1.6 * s) * size.width / 540;
    final turn = math.sin(age * (2.0 + 1.3 * s) + i).abs();
    canvas.save();
    canvas.translate(x, y);
    canvas.rotate(math.sin(age * 1.1 + i) * 1.2);
    canvas.drawOval(
      Rect.fromCenter(
        center: Offset.zero,
        width: 1.7 * w,
        height: w * (0.25 + 0.75 * turn),
      ),
      Paint()
        ..color =
            (i % 3 == 0 ? const Color(0xFFFFF1F4) : const Color(0xFFF4C6D2))
                .withValues(alpha: 0.86),
    );
    canvas.restore();
  }

  @override
  bool shouldRepaint(WorldParticles old) =>
      old.clockSeconds != clockSeconds || old.kind != kind;
}
