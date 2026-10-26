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
//   stream  water flowing towards the viewer (two-phase flow), only stream into stream
//   light   a lantern's flame flickering, and the light it throws
// The reference is the first prototype renderer (tuned 540 px wide); every
// distance here is in that reference's pixels times k = width / 540.
#include <flutter/runtime_effect.glsl>

precision highp float;

uniform vec2 uSize;     // the painting's rect on screen, px
uniform vec2 uOrigin;   // its top-left, px
uniform float uTime;    // seconds
uniform float uMotion;  // 1: alive; 0: reduce motion (the painting as painted)
uniform vec4 uOn;       // water, sky, grass, stream: 1 if the region exists
uniform vec4 uScene;    // trees (0/1), horizon (fraction; < 0 if none), unused, unused
uniform vec4 uLight;    // light x, y (fractions), size (fraction of the width), on (0/1)
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
  }
  if (uOn.z > 0.5 && reg.b > 0.001) {
    float gust = 0.55 + 0.45 * sin(TAU * t / 6.5);
    float wave = sin(TAU * (px.x / (210.0 * k) + px.y / (520.0 * k)) - TAU * t / 2.4);
    float fine = sin(TAU * (px.x / (37.0 * k) + px.y / (83.0 * k)) - TAU * t / 0.95);
    d.x += reg.b * (4.0 * gust * wave + 0.8 * fine) * k * m;
    d.y += reg.b * 0.8 * gust * wave * k * m;
    bright += reg.b * 0.07 * gust * wave * m;
  }
  if (uScene.x > 0.5) {
    float crown = pow(clamp(1.0 - uv.y / 0.62, 0.0, 1.0), 2.0) * (1.0 - reg.g);
    d.x += crown * 3.0 * k * m * (sin(TAU * t / 5.3 + px.x / (260.0 * k)) + 0.4 * sin(TAU * t / 2.1 + px.y / (90.0 * k)));
  }

  vec3 col = plate(px - d);
  if (uOn.y > 0.5 && reg.g > 0.001) {
    // Fading out to the left and right edges: it never reaches past them.
    float drift = 40.0 * k * m * sin(TAU * t / 90.0) * sin(3.14159265 * clamp(uv.x, 0.0, 1.0));
    vec2 from = vec2(px.x - drift, px.y);
    // Only sky drifts into the sky: where it would bring in a treetop, the sky stays as painted.
    float sky = smoothstep(0.5, 0.95, regionAt(from).g);
    col = mix(col, plate(from), reg.g * sky);
  }
  float stream = 1.0 - reg.a;
  if (uOn.w > 0.5 && stream > 0.001) {
    float period = 1.6;
    float p1 = fract(t / period);
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
