import 'package:flutter/foundation.dart';

/// One cut-out layer as the pipeline describes it (world.json, or the
/// server's `world` block, which adds `url` instead of `file`).
@immutable
class LayerSpec {
  const LayerSpec({
    required this.source,
    required this.depth,
    required this.motion,
    required this.anchor,
    this.name = '',
  });

  static LayerSpec? fromJson(Object? json) {
    if (json is! Map) return null;
    final source = json['url'] ?? json['file'];
    if (source is! String || source.isEmpty) return null;
    final a = json['anchor'];
    final anchor = a is List && a.length == 2 && a.every((v) => v is num)
        ? (x: (a[0] as num).toDouble().clamp(0.0, 1.0), y: (a[1] as num).toDouble().clamp(0.0, 1.0))
        : (x: 0.5, y: 1.0);
    return LayerSpec(
      source: source,
      depth: json['depth'] == 'near' ? 'near' : 'mid',
      motion: const {'sway', 'drift', 'float', 'still'}.contains(json['motion']) ? json['motion'] as String : 'still',
      anchor: anchor,
      name: json['name'] is String ? json['name'] as String : '',
    );
  }

  /// A file name (world.json) or a link (server block).
  final String source;
  final String depth;
  final String motion;

  /// Where the layer's motion pivots, as fractions of the painting (0..1).
  final ({double x, double y}) anchor;
  final String name;
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
  });

  static WorldManifest? fromJson(Object? json) {
    if (json is! Map || json['schema'] != 'elynd.world/1') return null;
    final size = json['size'];
    final aspect = size is List && size.length == 2 && size[0] is num && size[1] is num && (size[1] as num) > 0
        ? (size[0] as num) / (size[1] as num)
        : 9 / 16;
    return WorldManifest(
      title: json['title'] is String ? json['title'] as String : '',
      sleep: json['sleep'] == true,
      aspect: aspect.toDouble(),
      painting: json['painting'] as String? ?? 'painting.png',
      plate: json['plate'] as String?,
      layers: [
        for (final l in (json['layers'] is List ? json['layers'] as List : const [])) ?LayerSpec.fromJson(l),
      ],
    );
  }

  final String title;
  final bool sleep;

  /// Width / height of the painting.
  final double aspect;
  final String painting;
  final String? plate;
  final List<LayerSpec> layers;

  /// True if the world can move: a plate and at least one layer.
  bool get moving => plate != null && layers.isNotEmpty;
}
