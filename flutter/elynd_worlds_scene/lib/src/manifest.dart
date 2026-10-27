import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart' show ImageProvider;

import 'living.dart';

/// One cut-out layer as the pipeline describes it (world.json, or the
/// server's `world` block, which adds `url` instead of `file`).
@immutable
class LayerSpec {
  const LayerSpec({
    required this.source,
    required this.depth,
    required this.motion,
    required this.anchor,
    this.attached = 'ground',
    this.glows = false,
    this.name = '',
  });

  static LayerSpec? fromJson(Object? json) {
    if (json is! Map) return null;
    final source = json['url'] ?? json['file'];
    if (source is! String || source.isEmpty) return null;
    final a = json['anchor'];
    final anchor = a is List && a.length == 2 && a.every((v) => v is num)
        ? (
            x: (a[0] as num).toDouble().clamp(0.0, 1.0),
            y: (a[1] as num).toDouble().clamp(0.0, 1.0),
          )
        : (x: 0.5, y: 1.0);
    return LayerSpec(
      source: source,
      depth: json['depth'] == 'near' ? 'near' : 'mid',
      motion:
          const {
            'sway',
            'drift',
            'float',
            'flutter',
            'still',
          }.contains(json['motion'])
          ? json['motion'] as String
          : 'still',
      attached: const {'ground', 'edge', 'air'}.contains(json['attached'])
          ? json['attached'] as String
          : 'ground',
      glows: json['glows'] == true,
      anchor: anchor,
      name: json['name'] is String ? json['name'] as String : '',
    );
  }

  /// A file name (world.json) or a link (server block).
  final String source;
  final String depth;
  final String motion;

  /// How the object rests: "ground" (on land or water), "edge" or "air".
  final String attached;

  /// A light that glows by itself: it flickers.
  final bool glows;

  /// Where the layer's motion pivots, as fractions of the painting (0..1).
  final ({double x, double y}) anchor;
  final String name;
}

/// What moves in nature, as the pipeline saves it (world.json "living") or the
/// server sends it (with `regions_url` instead of the file name).
@immutable
class LivingSpec {
  const LivingSpec({
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

  static LivingSpec? fromJson(Object? json) {
    if (json is! Map) return null;
    final source = json['regions_url'] ?? json['regions'];
    final horizon = json['horizon'];
    final particles = json['particles'];
    final birds = json['birds'];
    final seed = json['seed'];
    return LivingSpec(
      regions: source is String && source.isNotEmpty ? source : null,
      water: json['water'] == true,
      sky: json['sky'] == true,
      grass: json['grass'] == true,
      stream: json['stream'] == true,
      trees: json['trees'] == true,
      horizon: horizon is num ? horizon.toDouble().clamp(0.0, 1.0) : null,
      lights: [
        for (final l
            in (json['lights'] is List ? json['lights'] as List : const []))
          ?WorldLight.fromJson(l),
      ],
      particles:
          const {
            'leaves',
            'snow',
            'pollen',
            'fireflies',
            'rain',
            'petals',
          }.contains(particles)
          ? particles as String
          : 'none',
      birds: const {'flock', 'gulls'}.contains(birds)
          ? birds as String
          : 'none',
      night: json['night'] == true,
      fish: json['fish'] == true,
      shadows: json['shadows'] == true,
      seed: seed is num ? seed.toDouble().clamp(0.0, 1.0) : 0,
    );
  }

  /// The region map: a file name (world.json) or a link (server).
  final String? regions;
  final bool water;
  final bool sky;
  final bool grass;
  final bool stream;
  final bool trees;
  final double? horizon;
  final List<WorldLight> lights;
  final String particles;

  /// What lives in the world (elynd_worlds/life.py): "flock" | "gulls" | "none" crossing the sky now and
  /// then; [night] stars twinkle and a shooting star passes; [fish] rings on calm water; [shadows] of
  /// clouds over the grass. [seed] makes one world's moments its own.
  final String birds;
  final bool night;
  final bool fish;
  final bool shadows;
  final double seed;

  /// Ready to draw, the region map loaded by [load].
  WorldLiving resolve(ImageProvider? Function(String source) load) =>
      WorldLiving(
        regions: regions == null ? null : load(regions!),
        water: water,
        sky: sky,
        grass: grass,
        stream: stream,
        trees: trees,
        horizon: horizon,
        lights: lights,
        particles: particles,
        birds: birds,
        night: night,
        fish: fish,
        shadows: shadows,
        seed: seed,
      );
}

/// A world as the pipeline saves it (schema "elynd.world/1").
@immutable
class WorldManifest {
  const WorldManifest({
    required this.title,
    required this.sleep,
    required this.aspect,
    required this.painting,
    required this.plate,
    required this.layers,
    this.focus = (x: 0.5, y: 0.5),
    this.living,
  });

  static WorldManifest? fromJson(Object? json) {
    if (json is! Map || json['schema'] != 'elynd.world/1') return null;
    final size = json['size'];
    final aspect =
        size is List &&
            size.length == 2 &&
            size[0] is num &&
            size[1] is num &&
            (size[1] as num) > 0
        ? (size[0] as num) / (size[1] as num)
        : 9 / 16;
    return WorldManifest(
      title: json['title'] is String ? json['title'] as String : '',
      sleep: json['sleep'] == true,
      aspect: aspect.toDouble(),
      painting: json['painting'] as String? ?? 'painting.png',
      plate: json['plate'] as String?,
      layers: [
        for (final l
            in (json['layers'] is List ? json['layers'] as List : const []))
          ?LayerSpec.fromJson(l),
      ],
      focus: point(json['focus']) ?? (x: 0.5, y: 0.5),
      living: LivingSpec.fromJson(json['living']),
    );
  }

  /// A [x, y] pair of fractions, or null.
  static ({double x, double y})? point(Object? json) =>
      json is List && json.length == 2 && json.every((v) => v is num)
      ? (
          x: (json[0] as num).toDouble().clamp(0.0, 1.0),
          y: (json[1] as num).toDouble().clamp(0.0, 1.0),
        )
      : null;

  final String title;
  final bool sleep;

  /// Width / height of the painting.
  final double aspect;
  final String painting;
  final String? plate;
  final List<LayerSpec> layers;

  /// Where the subjects are: screens crop around it.
  final ({double x, double y}) focus;

  /// What moves in nature; null for worlds made before it existed.
  final LivingSpec? living;

  /// True if the world can move: a plate and at least one layer.
  bool get moving => plate != null && layers.isNotEmpty;
}
