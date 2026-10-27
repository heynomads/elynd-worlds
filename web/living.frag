#version 460 core
// ELYND Worlds: the living painting. Moves each part of the painting as it moves
// in nature, read from the region map SAM 3.1 made (red = water, green = sky,
// blue = grass, alpha = 1 - stream; inverted so the premultiplied picture keeps
// the other three wherever no stream flows):
//   water   ripples travelling towards the viewer (shorter far away), reflections
//           wobbling sideways, now and then a tiny sparkle of sunlight on the
//           bright parts, each at its own place (never a pattern)
//   grass   waves of wind rolling through it, the light catching the bent grass
//   trees   a light wind in the crowns, strongest at the top, never the sky
//   sky     the clouds drifting slowly to one side and back, less towards the
//           painting's left and right edges (no paint is ever needed beyond them);
//           only sky drifts into the sky, never a treetop or a hill beside it
//   stream  water flowing towards the viewer (two-phase flow, each part on its own
//           phase so it never pulses), only stream into stream
//   light   a lantern's flame flickering, and the light it throws
// and what lives there (life.py decides which belong to a world; all subtle, all
// only where they belong in the region map):
//   birds   now and then a small flock crossing the upper sky, flapping and
//           gliding, or one to three gulls gliding over the sea; only over sky
//           (they pass behind tree crowns), never at night
//   night   the painted stars twinkle; now and then a faint shooting star
//   fish    on calm water, now and then the rings of a fish rising
//   rain    drops ringing on the water (the streaks are drawn above the painting)
//   shadows clouds' shadows sliding slowly over the grass
// The reference is the first prototype renderer (tuned 540 px wide); every
// distance here is in that reference's pixels times k = width / 540.
#include <flutter/runtime_effect.glsl>

precision highp float;

uniform vec2 uSize;     // the painting's rect on screen, px
uniform vec2 uOrigin;   // its top-left, px
uniform float uTime;    // seconds
uniform float uMotion;  // 1: alive; 0: reduce motion (the painting as painted)
uniform vec4 uOn;       // water, sky, grass, stream: 1 if the region exists
uniform vec4 uScene;    // trees (0/1), horizon (fraction; < 0 if none), rain (0/1), seed [0, 1)
uniform vec4 uLight;    // light x, y (fractions), size (fraction of the width), on (0/1)
uniform vec4 uLife;     // birds (0 none, 1 flock, 2 gulls), night (0/1), fish (0/1), cloud shadows (0/1)
uniform sampler2D uPlate;
uniform sampler2D uRegions;

out vec4 fragColor;

const float TAU = 6.28318530718;

vec3 plate(vec2 px) {
  return texture(uPlate, clamp(px / uSize, vec2(0.0005), vec2(0.9995))).rgb;
}

vec4 regionAt(vec2 px) {
  return texture(uRegions, clamp(px / uSize, vec2(0.0005), vec2(0.9995)));
}

// A number in [0, 1) for each cell of a grid, the same on every device (a hash
// without sine, Dave Hoskins).
float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// Smooth noise in [0, 1): between the hashed corners of a grid, eased.
float noise(vec2 q) {
  vec2 i = floor(q);
  vec2 f = fract(q);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

// Sunlight sparkling on the water: in some cells of a grid (q in reference
// pixels) a tiny glint, flat as the water, at its own place, flashing for a
// moment every few seconds; larger near the viewer (p: 0 at the horizon, 1 at
// the bottom). One cell per pixel: each glint stays well inside its cell.
float glitter(vec2 q, float p, float t) {
  vec2 cell = vec2(24.0, 9.0);
  vec2 c = floor(q / cell);
  if (hash(c) > 0.4) return 0.0; // most places never sparkle
  vec2 at = (c + 0.3 + 0.4 * vec2(hash(c + 17.3), hash(c + 41.9))) * cell;
  float period = 2.4 + 2.2 * hash(c + 73.1);
  float flash = pow(max(sin(TAU * (t / period + hash(c + 5.7))), 0.0), 30.0);
  vec2 d = (q - at) / vec2(1.6 + 2.6 * p, 0.6 + 0.6 * p);
  return flash * exp(-dot(d, d));
}

// ─── What lives in the world ───────────────────────────────────────────

// Distance from p to the segment a-b.
float segment(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  return length(pa - ba * h);
}

// Distance from p to a capsule from a to b whose radius goes from ra (at a) to rb (at b).
float taper(vec2 p, vec2 a, vec2 b, float ra, float rb) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  return length(pa - ba * h) - mix(ra, rb, h);
}

// One bird seen from below and far away, in units of its half-span: a small body and two wings, each
// broad at the shoulder, slimmer at the wrist and pointed at the tip, raised by `lift` (radians; the
// down-stroke below 0) and bent at the wrist by `bend` (a gull's M). How much of it covers q (0-1), with
// an edge `aa` wide.
float bird(vec2 q, float lift, float bend, float aa) {
  vec2 s = vec2(abs(q.x), q.y);                      // both wings are mirror images
  vec2 shoulder = vec2(0.06, 0.0);
  vec2 wrist = shoulder + 0.42 * vec2(cos(lift), sin(lift));
  vec2 tip = wrist + 0.56 * vec2(cos(lift - bend), sin(lift - bend));
  float wing = min(taper(s, shoulder, wrist, 0.1, 0.065), taper(s, wrist, tip, 0.065, 0.008));
  float body = (length(q / vec2(0.12, 0.1)) - 1.0) * 0.1;
  return 1.0 - smoothstep(0.0, aa, min(wing, body));
}

// Birds crossing the sky now and then: a small flock (kind 1) or gulls (kind 2). Returns their cover at px
// (0-1); `tone` is how dark they are against this sky (0 dark silhouette, 1 a white gull lit by the sun).
float birds(vec2 px, float k, float t, float kind, float seed, out float tone) {
  tone = 0.0;
  float gulls = step(1.5, kind);
  float period = mix(70.0, 95.0, gulls);
  float slot = floor(t / period);
  float h = hash(vec2(slot + 1.0, seed * 97.0 + 3.0));
  float start = slot * period + 6.0 + 22.0 * h;        // when this pass begins
  float crossing = mix(14.0, 34.0, gulls) * (0.85 + 0.3 * hash(vec2(slot, 7.0 + seed)));
  float age = t - start;
  if (age < 0.0 || age > crossing) return 0.0;
  float dir = hash(vec2(slot, 11.0 + seed)) > 0.5 ? 1.0 : -1.0;
  float alt = (0.09 + 0.2 * hash(vec2(slot, 23.0 + seed))) * uSize.y;   // the upper sky
  float n = mix(4.0 + floor(5.0 * hash(vec2(slot, 31.0))), 1.0 + floor(3.0 * hash(vec2(slot, 37.0))), gulls);
  float progress = age / crossing;
  float cover = 0.0;
  for (int i = 0; i < 9; i++) {
    float fi = float(i);
    if (fi >= n) break;
    vec2 r = vec2(hash(vec2(fi, slot + 41.0)), hash(vec2(fi, slot + 53.0)));
    // A loose flock: each bird a little ahead or behind, above or below, a little smaller far away.
    float span = mix(4.5 + 3.5 * r.y, 9.5 + 3.5 * r.y, gulls) * k;
    // Spread by their order, then a little each way: a loose flock, never two birds on top of each other.
    float x0 = mix(-0.12, 1.12, progress) + (fi - 0.5 * (n - 1.0)) * mix(0.032, 0.1, gulls)
             + (r.x - 0.5) * mix(0.024, 0.05, gulls);
    float x = (dir > 0.0 ? x0 : 1.0 - x0) * uSize.x;
    float bob = sin(TAU * (t / mix(1.6, 6.5, gulls)) + fi * 1.7) * mix(1.6, 4.0, gulls) * k;
    float y = alt + (r.y - 0.5) * mix(0.05, 0.07, gulls) * uSize.y + bob;
    vec2 q = (px - vec2(x, y)) / span;
    if (dot(q, q) > 2.0) continue;
    q.x *= dir;                                          // facing its way (the shape is symmetric anyway)
    q.y = -q.y;                                          // screen y grows downwards
    // Wingbeats in bursts, then a glide: a small bird beats fast and glides briefly, a gull mostly glides.
    float phase = t + 13.0 * r.x;
    float cycle = mix(2.2, 7.0, gulls);
    float beating = step(fract(phase / cycle), mix(0.62, 0.3, gulls));
    float beat = sin(TAU * phase * mix(5.5, 2.1, gulls));
    float lift = mix(mix(0.32, 0.12, gulls), 0.18 + 0.55 * beat, beating);
    float bend = mix(0.35, 0.75, gulls) + 0.15 * beat * beating;
    cover = max(cover, bird(q, lift, bend, 1.4 / max(span, 1.0)));
  }
  tone = gulls;
  // Fading in at the start and out at the end of the pass, never popping into view.
  return cover * smoothstep(0.0, 0.06, progress) * smoothstep(1.0, 0.94, progress);
}

// Rings on the water: a fish rising now and then (two rings spreading slowly), or raindrops (small, many).
// A ring is a crest and a trough: it bends the reflection (the displacement returned) and, tilting the
// water, shows a little more sky on one side and a little less on the other (`shade`, signed), which is
// what makes a ring visible on bright and on dark water alike.
vec2 rings(vec2 px, float k, float t, float horizon, float drops, out float shade) {
  shade = 0.0;
  vec2 d = vec2(0.0);
  if (drops > 0.5) {
    // Many small rings: a grid laid on the water's surface (so smaller and denser far away), each cell
    // with a drop now and then. A grid scaled on the screen instead stretches every ring tall (a cell's
    // size would change within the cell). On the surface, one px is one px on the screen at the bottom:
    // w.x across (from the middle), w.y away from us (d w.y / d y = 1 / scale; above the horizon, as at it).
    float span = uSize.y * max(1.0 - horizon, 0.001);
    float pu = (px.y / uSize.y - horizon) / max(1.0 - horizon, 0.001);
    float s = 0.55 + 0.45 * clamp(pu, 0.0, 1.0);
    float far = log(0.55) / 0.45;
    vec2 w = vec2((px.x - 0.5 * uSize.x) / s, span * (pu >= 0.0 ? log(s) / 0.45 : far + pu / 0.55));
    vec2 cell = vec2(24.0, 9.0) * k;
    vec2 g = w / cell;
    vec2 side = step(0.5, fract(g)) * 2.0 - 1.0;
    // A ring spreads past its cell's edge: its own cell and the three nearest, never further.
    for (int a = 0; a < 2; a++) {
      for (int b = 0; b < 2; b++) {
        vec2 c = floor(g) + vec2(float(a), float(b)) * side;
        float period = 0.7 + 1.0 * hash(c + 3.1);
        float age = fract(t / period + hash(c + 9.7)) * period;
        vec2 dw = (c + 0.2 + 0.6 * vec2(hash(c + 1.3), hash(c + 5.9))) * cell;   // the drop, on the surface
        float pa = dw.y >= span * far ? (exp(0.45 * dw.y / span) - 0.55) / 0.45 : (dw.y / span - far) * 0.55;
        float near = clamp(pa, 0.0, 1.0);
        vec2 at = vec2(0.5 * uSize.x + dw.x * (0.55 + 0.45 * near), uSize.y * horizon + span * pa);
        vec2 q = (px - at) / vec2(1.0, 0.26 + 0.24 * near);   // seen at a low angle: far rings are flatter
        float r = (0.6 + 12.0 * age) * k * (0.45 + 0.55 * near);
        float x = (length(q) - r) / (1.1 * k * (0.5 + 0.5 * near));
        float wave = exp(-x * x) * cos(2.4 * x) * exp(-3.6 * age);
        shade += 0.24 * wave;
        d += normalize(q + 1e-4) * wave * 1.3 * k;
      }
    }
    return d;
  }
  float slot = floor(t / 11.0);
  for (int j = 0; j < 2; j++) {                          // this rise and the one before, still spreading
    float s = slot - float(j);
    if (hash(vec2(s, 61.0)) > 0.75) continue;            // now and then, not like a clock
    float age = t - (s * 11.0 + 9.0 * hash(vec2(s, 67.0)));
    if (age < 0.0 || age > 4.5) continue;
    float y = mix(horizon + 0.06, min(horizon + 0.5, 0.86), hash(vec2(s, 71.0)));
    vec2 at = vec2(mix(0.18, 0.82, hash(vec2(s, 73.0))), y) * uSize;
    if (regionAt(at).r < 0.6) continue;                  // only where it is water
    float pa = clamp((y - horizon) / max(1.0 - horizon, 0.001), 0.0, 1.0);
    vec2 q = (px - at) / vec2(1.0, 0.26 + 0.24 * pa);
    float len = length(q);
    for (int n = 0; n < 2; n++) {
      float r = (3.0 + 15.0 * age - 10.0 * float(n)) * k * (0.45 + 0.55 * pa);
      if (r <= 0.0) continue;
      float x = (len - r) / (2.2 * k * (0.5 + 0.5 * pa));
      float wave = exp(-x * x) * cos(2.4 * x) * exp(-0.75 * age) * (1.0 - 0.35 * float(n));
      shade += 0.13 * wave;
      d += normalize(q + 1e-4) * wave * 2.4 * k;
    }
  }
  return d;
}

// A shooting star now and then on a night sky: a short bright streak with a fading tail.
float shootingStar(vec2 px, float k, float t, float seed) {
  float period = 47.0;
  float slot = floor(t / period);
  if (hash(vec2(slot, 83.0 + seed)) > 0.6) return 0.0;
  float age = t - (slot * period + 5.0 + 30.0 * hash(vec2(slot, 89.0)));
  if (age < 0.0 || age > 0.8) return 0.0;
  vec2 from = vec2(mix(0.15, 0.85, hash(vec2(slot, 97.0))), mix(0.06, 0.26, hash(vec2(slot, 101.0)))) * uSize;
  float side = hash(vec2(slot, 103.0)) > 0.5 ? 1.0 : -1.0;
  vec2 dir = normalize(vec2(side, 0.45));
  vec2 head = from + dir * (age / 0.8) * 120.0 * k;
  float along = dot(px - head, -dir);
  float across = abs(dot(px - head, vec2(-dir.y, dir.x)));
  float tail = 36.0 * k;
  if (along < -1.5 * k || along > tail) return 0.0;
  float glow = (1.0 - clamp(along / tail, 0.0, 1.0)) * exp(-pow(across / (0.8 * k), 2.0));
  return glow * sin(3.14159265 * age / 0.8);           // in and out, never a sudden flash
}

void main() {
  vec2 px = FlutterFragCoord().xy - uOrigin;
  vec2 uv = px / uSize;
  vec4 reg = regionAt(px);
  float k = uSize.x / 540.0;
  float t = uTime;
  float m = uMotion;
  vec2 d = vec2(0.0);
  float bright = 1.0;
  float glint = 0.0;

  if (uOn.x > 0.5 && reg.r > 0.001) {
    float horizon = uScene.y >= 0.0 ? uScene.y : 0.5;
    float p = clamp((uv.y - horizon) / max(1.0 - horizon, 0.001), 0.0, 1.0);
    float wavelength = (10.0 + 46.0 * p) * k;
    float phase = TAU * (px.y / wavelength) - TAU * t / 2.6;
    float across = TAU * px.x / (230.0 * k) + TAU * t / 7.0;
    float amp = (0.8 + 4.6 * p) * k * m;
    d.x += reg.r * amp * (sin(phase) + 0.35 * sin(across + px.y / (37.0 * k)));
    d.y += reg.r * 0.35 * amp * sin(phase * 0.7 + 1.3);
    glint = reg.r * glitter((px - d) / k, p, t) * m; // the glints ride the ripples
    if (uLife.z > 0.5 || uScene.z > 0.5) {
      float shade;
      d += reg.r * m * rings(px, k, t, horizon, uScene.z, shade);
      bright += reg.r * m * shade;
    }
  }
  if (uOn.z > 0.5 && reg.b > 0.001) {
    float gust = 0.55 + 0.45 * sin(TAU * t / 6.5);
    float wave = sin(TAU * (px.x / (210.0 * k) + px.y / (520.0 * k)) - TAU * t / 2.4);
    float fine = sin(TAU * (px.x / (37.0 * k) + px.y / (83.0 * k)) - TAU * t / 0.95);
    d.x += reg.b * (4.0 * gust * wave + 0.8 * fine) * k * m;
    d.y += reg.b * 0.8 * gust * wave * k * m;
    bright += reg.b * 0.07 * gust * wave * m;
    if (uLife.w > 0.5) {
      // Clouds' shadows: large soft patches drifting slowly across the meadow.
      vec2 q = (px / k + vec2(-9.0, 2.0) * t) / 300.0;
      float c = 0.65 * noise(q + uScene.w * 17.0) + 0.35 * noise(2.1 * q + 3.7);
      bright -= reg.b * 0.12 * smoothstep(0.52, 0.72, c) * m;
    }
  }
  if (uScene.x > 0.5) {
    float crown = pow(clamp(1.0 - uv.y / 0.62, 0.0, 1.0), 2.0) * (1.0 - reg.g);
    d.x += crown * 3.0 * k * m * (sin(TAU * t / 5.3 + px.x / (260.0 * k)) + 0.4 * sin(TAU * t / 2.1 + px.y / (90.0 * k)));
  }

  vec3 col = plate(px - d);
  vec2 seen = px;  // where the sky seen here is in the painting (it drifts)
  if (uOn.y > 0.5 && reg.g > 0.001) {
    // Fading out to the left and right edges: it never reaches past them.
    float drift = 40.0 * k * m * sin(TAU * t / 90.0) * sin(3.14159265 * clamp(uv.x, 0.0, 1.0));
    vec2 from = vec2(px.x - drift, px.y);
    // Only sky drifts into the sky: where it would bring in a treetop, the sky stays as painted.
    float sky = smoothstep(0.5, 0.95, regionAt(from).g);
    col = mix(col, plate(from), reg.g * sky);
    seen = reg.g * sky > 0.5 ? from : px;
  }
  if (uLife.y > 0.5 && uOn.y > 0.5 && reg.g > 0.001) {
    // The painted stars twinkle: a point brighter than all the sky around it (never the rim of a moon or
    // a cloud, bright on one side), each star on its own rhythm wherever the drift takes it.
    const vec3 third = vec3(0.33333);
    float around = max(max(dot(plate(seen + vec2(3.0, 0.0) * k), third), dot(plate(seen - vec2(3.0, 0.0) * k), third)),
                       max(dot(plate(seen + vec2(0.0, 3.0) * k), third), dot(plate(seen - vec2(0.0, 3.0) * k), third)));
    float star = smoothstep(0.08, 0.22, dot(plate(seen), third) - around);
    vec2 cell = floor(seen / (6.0 * k));
    float twinkle = 0.82 + 0.18 * sin(TAU * t / (1.7 + 2.3 * hash(cell)) + TAU * hash(cell + 3.3));
    col *= mix(1.0, twinkle, star * reg.g * m);
    col = mix(col, vec3(0.97, 0.98, 1.0), 0.7 * shootingStar(px, k, t, uScene.w) * reg.g * m);
  }
  if (uLife.x > 0.5 && uOn.y > 0.5) {
    float tone;
    // Over the sky only (they pass behind a tree's crown, never in front of a hill): the map's sky,
    // softly, so that a bird never thins out over a cloud.
    float cover = birds(px, k, t, uLife.x, uScene.w, tone) * smoothstep(0.12, 0.45, reg.g) * m;
    if (cover > 0.0) {
      float skyLum = dot(col, vec3(0.33333));
      // A small bird is a dark silhouette; a gull is white, darker against a bright sky than a dim one.
      vec3 ink = tone > 0.5 ? vec3(mix(0.93, 0.45, smoothstep(0.5, 0.9, skyLum))) : vec3(0.09, 0.09, 0.11);
      col = mix(col, ink, mix(0.72, 0.85, step(0.5, tone)) * cover);   // a little air between us and them
    }
  }
  float stream = 1.0 - reg.a;
  if (uOn.w > 0.5 && stream > 0.001) {
    float period = 1.6;
    // Each part of the stream runs on its own phase (smooth noise): the two
    // samples' crossfade never blurs and sharpens the whole stream at once,
    // which read as a pulse.
    float p1 = fract(t / period + noise(px / (48.0 * k)));
    float p2 = fract(p1 + 0.5);
    vec2 v = vec2(-4.0, 20.0) * k * m;
    float rx = 1.2 * k * m * sin(TAU * px.y / (14.0 * k) - TAU * t / 0.9);
    vec2 a1 = vec2(px.x + rx, px.y) - v * p1 * period;
    vec2 a2 = vec2(px.x + rx, px.y) - v * p2 * period;
    // Only the stream flows into the stream: a bank or a stone upstream stays where it is.
    vec3 s1 = mix(col, plate(a1), smoothstep(0.5, 0.95, 1.0 - regionAt(a1).a));
    vec3 s2 = mix(col, plate(a2), smoothstep(0.5, 0.95, 1.0 - regionAt(a2).a));
    float w1 = 1.0 - abs(2.0 * p1 - 1.0);
    col = mix(col, s1 * w1 + s2 * (1.0 - w1), stream);
  }

  col *= bright;
  // A glint only where the water is bright (the sun's or the sky's reflection).
  float lum = dot(col, vec3(0.33333));
  col = mix(col, vec3(1.0, 0.96, 0.88), clamp(glint * clamp((lum - 0.55) / 0.25, 0.0, 1.0), 0.0, 0.85));

  if (uLight.w > 0.5) {
    float n = sin(TAU * 1.7 * t) * 0.5 + sin(TAU * 2.9 * t + 1.1) * 0.3 + sin(TAU * 5.3 * t + 2.3) * 0.2;
    float size = max(uLight.z * uSize.x, 1.0);
    float r = distance(px, uLight.xy * uSize);
    float fall = exp(-pow(r / (size * 2.2), 2.0));
    col *= vec3(1.0) + 0.11 * n * fall * m * vec3(1.0, 0.86, 0.62);
  }
  fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
