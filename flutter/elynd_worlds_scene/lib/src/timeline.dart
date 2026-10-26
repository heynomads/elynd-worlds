import 'dart:math' as math;

import 'package:flutter/foundation.dart';

/// One silence of the voice, in seconds (a session's `silence_regions`).
@immutable
class WorldSilence {
  const WorldSilence(this.start, this.end);

  /// From a `silence_regions` entry: {"start_seconds", "end_seconds"}. Null if malformed.
  static WorldSilence? fromJson(Object? json) {
    if (json is! Map) return null;
    final start = json['start_seconds'], end = json['end_seconds'];
    if (start is! num || end is! num || end <= start) return null;
    return WorldSilence(start.toDouble(), end.toDouble());
  }

  final double start;
  final double end;
}

/// A moment the voice says "breathe in" or "breathe out".
@immutable
class WorldCue {
  const WorldCue(this.at, {required this.inhale});

  final double at;
  final bool inhale;
}

/// When the world breathes and how much of it shows, one keyframe per second.
///
/// The same rules as `elynd_worlds/timeline.py`, frame for frame (the parity
/// test compares this class with the Python output):
///
/// * visibility: the whole world while the listener arrives (45 s), fading to a
///   glow while their eyes are closed, rising in silences of 30 s or more and
///   over the last 30 s. A sleep world stays a dim glow and ends dark.
/// * breath: a slow wave, ~5.5 breaths a minute in silences and gentler while
///   the voice speaks; a spoken cue makes the world inhale over 4 s or exhale
///   over 6 s exactly when the voice says so.
@immutable
class WorldTimeline {
  const WorldTimeline({required this.breath, required this.visibility});

  /// A still, fully visible world.
  factory WorldTimeline.still() => const WorldTimeline(breath: [0.5], visibility: [1.0]);

  factory WorldTimeline.build({
    required double durationSeconds,
    List<WorldSilence> silences = const [],
    List<WorldCue> cues = const [],
    bool sleep = false,
  }) {
    final sorted = [...cues]..sort((a, b) => a.at.compareTo(b.at));
    final breath = <double>[];
    final visibility = <double>[];
    for (var second = 0; second <= durationSeconds.ceil(); second++) {
      final t = second.toDouble();
      breath.add(_round3(_breathAt(t, silences, sorted).clamp(0.0, 1.0)));
      visibility.add(_round3(_visibilityAt(t, durationSeconds, silences, sleep).clamp(0.0, 1.0)));
    }
    return WorldTimeline(breath: breath, visibility: visibility);
  }

  static const resonantPeriod = 60 / 5.5;
  static const speechPeriod = 10.0;
  static const inhaleSeconds = 4.0;
  static const exhaleSeconds = 6.0;
  static const arriveSeconds = 45.0;
  static const fadeSeconds = 20.0;
  static const returnSeconds = 30.0;
  static const longSilence = 30.0;
  static const glow = 0.25;
  static const sleepGlow = 0.15;

  final List<double> breath;
  final List<double> visibility;

  static WorldSilence? _silenceAt(List<WorldSilence> silences, double t) {
    for (final s in silences) {
      if (s.start <= t && t < s.end) return s;
    }
    return null;
  }

  static double _breathAt(double t, List<WorldSilence> silences, List<WorldCue> cues) {
    for (final cue in cues) {
      if (cue.inhale && cue.at <= t && t < cue.at + inhaleSeconds) return (t - cue.at) / inhaleSeconds;
      if (!cue.inhale && cue.at <= t && t < cue.at + exhaleSeconds) return 1.0 - (t - cue.at) / exhaleSeconds;
    }
    final quiet = _silenceAt(silences, t) != null;
    final period = quiet ? resonantPeriod : speechPeriod;
    final depth = quiet ? 0.8 : 0.3;
    return 0.5 + 0.5 * depth * math.sin(2 * math.pi * t / period - math.pi / 2);
  }

  static double _visibilityAt(double t, double duration, List<WorldSilence> silences, bool sleep) {
    double v;
    if (t < arriveSeconds) {
      v = 1.0;
    } else if (t < arriveSeconds + fadeSeconds) {
      v = 1.0 - (1.0 - glow) * (t - arriveSeconds) / fadeSeconds;
    } else {
      v = glow;
    }
    final silence = _silenceAt(silences, t);
    if (silence != null && silence.end - silence.start >= longSilence) {
      final ramp = math.min(math.min(t - silence.start, silence.end - t), 8.0) / 8.0;
      v = math.max(v, glow + 0.5 * ramp);
    }
    final left = duration - t;
    if (left < returnSeconds) v = math.max(v, 1.0 - (1.0 - glow) * left / returnSeconds);
    if (sleep && t >= arriveSeconds) {
      v = math.min(v, sleepGlow);
      if (left < returnSeconds) v = sleepGlow * math.max(left, 0.0) / returnSeconds;
    }
    return v;
  }

  static double _round3(double x) => (x * 1000).round() / 1000;

  static double _sample(List<double> values, double seconds) {
    if (values.isEmpty) return 0;
    if (seconds <= 0) return values.first;
    final i = seconds.floor();
    if (i >= values.length - 1) return values.last;
    final f = seconds - i;
    return values[i] * (1 - f) + values[i + 1] * f;
  }

  double breathAt(double seconds) => _sample(breath, seconds);
  double visibilityAt(double seconds) => _sample(visibility, seconds);
}
