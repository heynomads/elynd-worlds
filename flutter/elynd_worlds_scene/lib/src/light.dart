import 'dart:ui' show Color;

import 'package:flutter/foundation.dart';

/// How a world's light changes over a meditation. It starts exactly as painted
/// and changes so slowly that it shows when the listener opens their eyes at
/// the end: time has passed in the place too.
///
/// * [sunrise]: a dawn or morning world warms and brightens as the sun climbs.
/// * [deepen]: a golden-hour or dusk world's glow deepens and warms.
/// * [nightfall]: a sleep world dims toward night and keeps its light warm: before sleep, light should be low,
///   and warm rather than blue (Brown et al. 2022, PLoS Biology; Schöllhorn et al. 2023).
/// * [none]: midday and night worlds stay as painted.
enum LightJourney {
  none,
  sunrise,
  deepen,
  nightfall;

  /// The light of a world painted at [timeOfDay] (the design's words, any
  /// case), for a sleep meditation or not.
  static LightJourney of({String? timeOfDay, bool sleep = false}) {
    if (sleep) return nightfall;
    final t = (timeOfDay ?? '').toLowerCase();
    if (t.contains('dawn') || t.contains('sunrise') || t.contains('morning')) {
      return sunrise;
    }
    if (const ['golden', 'sunset', 'dusk', 'evening', 'twilight', 'afternoon']
        .any(t.contains)) {
      return deepen;
    }
    return none;
  }

  /// How far the light has come [seconds] into a session of [duration]
  /// seconds: 0 at the start, 1 at the end, easing in and out.
  static double progress(double seconds, double duration) {
    if (duration <= 0) return 0;
    final p = (seconds / duration).clamp(0.0, 1.0);
    return p * p * (3 - 2 * p);
  }

  /// At the journey's end: how much of the picture becomes the light ([mix]), that light's colour (red, green,
  /// blue, 0 to 1), and the picture's own brightness ([exposure], at most 1). The picture is blended toward the
  /// light rather than scaled up, so nothing ever passes white: the brightest clouds keep their detail.
  ({double mix, double r, double g, double b, double exposure}) get _end => switch (this) {
    LightJourney.sunrise => (mix: 0.16, r: 1.00, g: 0.86, b: 0.64, exposure: 1.0),
    LightJourney.deepen => (mix: 0.20, r: 0.95, g: 0.60, b: 0.32, exposure: 0.97),
    LightJourney.nightfall => (mix: 0.18, r: 0.30, g: 0.16, b: 0.06, exposure: 0.72),
    LightJourney.none => (mix: 0, r: 0, g: 0, b: 0, exposure: 1),
  };

  /// The colour matrix ([ColorFilter.matrix]: its last column in 0 to 255) at [progress] (0: as painted), with
  /// [opacity] as its alpha: one layer does both.
  List<double> matrix(double progress, {double opacity = 1}) {
    final p = progress.clamp(0.0, 1.0);
    final end = _end;
    final k = end.mix * p;
    final gain = (1 + (end.exposure - 1) * p) * (1 - k);
    return [
      gain, 0, 0, 0, end.r * k * 255, //
      0, gain, 0, 0, end.g * k * 255, //
      0, 0, gain, 0, end.b * k * 255, //
      0, 0, 0, opacity, 0,
    ];
  }

  /// The same light as two coats painted over the scene, so it needs no layer of its own: a black coat whose
  /// alpha is [shade] brings the picture down to the light's gain, then [add], painted with BlendMode.plus,
  /// adds the light's colour. Over the opaque scene this is exactly [matrix] (light_test.dart).
  ({double shade, Color add}) coats(double progress) {
    final p = progress.clamp(0.0, 1.0);
    final end = _end;
    final k = end.mix * p;
    final gain = (1 + (end.exposure - 1) * p) * (1 - k);
    return (shade: 1 - gain, add: Color.from(alpha: 1, red: end.r * k, green: end.g * k, blue: end.b * k));
  }

  @visibleForTesting
  static List<double> identity({double opacity = 1}) => [
    1, 0, 0, 0, 0, //
    0, 1, 0, 0, 0, //
    0, 0, 1, 0, 0, //
    0, 0, 0, opacity, 0,
  ];
}
