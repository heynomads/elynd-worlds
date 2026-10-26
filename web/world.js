// ELYND Worlds in the browser: the same timeline, motion and living painting
// as the phone (flutter/elynd_worlds_scene: timeline.dart, motion.dart,
// living.dart and shaders/living.frag, which this page compiles for WebGL 2
// as it is) and the engine (elynd_worlds/timeline.py). web/parity.html checks
// this timeline against the Python one, frame for frame.
"use strict";

const WorldTimeline = (() => {
  const RESONANT = 60 / 5.5, SPEECH = 10, INHALE = 4, EXHALE = 6;
  const ARRIVE = 45, FADE = 20, RETURN = 30, LONG_SILENCE = 30, GLOW = 0.25, SLEEP_GLOW = 0.15;
  // Python's round(v, 3): the exact binary value of v, ties to even. Not
  // Math.round(v * 1000): the product itself rounds (0.88749999… × 1000 is
  // 887.5 in floating point), which would put a keyframe 0.001 off Python's.
  const bits = new DataView(new ArrayBuffer(8));
  function round3(v) {
    if (v === 0) return 0;
    bits.setFloat64(0, v);
    const raw = bits.getBigUint64(0);
    const exponent = Number((raw >> 52n) & 0x7ffn);
    let mantissa = raw & 0xfffffffffffffn;
    let power = -1074;
    if (exponent !== 0) { mantissa |= 1n << 52n; power = exponent - 1075; }
    const [num, den] = power >= 0 ? [mantissa << BigInt(power), 1n] : [mantissa, 1n << BigInt(-power)];
    let q = (num * 1000n) / den;
    const twice = ((num * 1000n) % den) * 2n;
    if (twice > den || (twice === den && q % 2n === 1n)) q += 1n;
    return Number(q) / 1000;
  }
  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  const silenceAt = (silences, t) => silences.find(([s, e]) => s <= t && t < e) || null;

  // cues: [{at, inhale}], sorted by `at`
  function breathAt(t, silences, cues) {
    for (const cue of cues) {
      if (cue.inhale && cue.at <= t && t < cue.at + INHALE) return (t - cue.at) / INHALE;
      if (!cue.inhale && cue.at <= t && t < cue.at + EXHALE) return 1 - (t - cue.at) / EXHALE;
    }
    const quiet = silenceAt(silences, t) !== null;
    const period = quiet ? RESONANT : SPEECH;
    const depth = quiet ? 0.8 : 0.3;
    return 0.5 + 0.5 * depth * Math.sin((2 * Math.PI * t) / period - Math.PI / 2);
  }

  function visibilityAt(t, duration, silences, sleep) {
    let v;
    if (t < ARRIVE) v = 1;
    else if (t < ARRIVE + FADE) v = 1 - ((1 - GLOW) * (t - ARRIVE)) / FADE;
    else v = GLOW;
    const silence = silenceAt(silences, t);
    if (silence && silence[1] - silence[0] >= LONG_SILENCE) {
      const ramp = Math.min(Math.min(t - silence[0], silence[1] - t), 8) / 8;
      v = Math.max(v, GLOW + 0.5 * ramp);
    }
    const left = duration - t;
    if (left < RETURN) v = Math.max(v, 1 - ((1 - GLOW) * left) / RETURN);
    if (sleep && t >= ARRIVE) {
      v = Math.min(v, SLEEP_GLOW);
      if (left < RETURN) v = (SLEEP_GLOW * Math.max(left, 0)) / RETURN;
    }
    return v;
  }

  /** One keyframe per second, as the phone builds it. */
  function build(duration, silences = [], cues = [], sleep = false) {
    const sorted = [...cues].sort((a, b) => a.at - b.at);
    const breath = [], visibility = [];
    for (let second = 0; second <= Math.ceil(duration); second++) {
      breath.push(round3(clamp01(breathAt(second, silences, sorted))));
      visibility.push(round3(clamp01(visibilityAt(second, duration, silences, sleep))));
    }
    return { breath, visibility };
  }

  /** A keyframe list read at `t` seconds, linearly between keyframes. */
  function at(frames, t) {
    if (!frames.length) return 0;
    const last = frames.length - 1;
    if (t <= 0) return frames[0];
    if (t >= last) return frames[last];
    const i = Math.floor(t);
    return frames[i] + (frames[i + 1] - frames[i]) * (t - i);
  }

  return { build, at, ARRIVE, GLOW };
})();

// How a world's objects move: as such things move in nature, as the phone
// moves them (motion.dart). A boat sinks a little and rises back to where it
// is painted, and rolls; something in the air floats on a slow loop; a
// butterfly also folds its wings now and then; plants sway in gusts; mist
// drifts; stones, posts, cabins and lanterns stay still (a lantern flickers).
// Offsets are fractions of the painting's height.
const WorldMotion = {
  bob: 0.006,
  bobRoll: (0.8 * Math.PI) / 180,
  hoverX: 0.010,
  hoverY: 0.007,
  wingsFolded: 0.45,
  wingCycle: 4.2,
  wingFold: 0.55,
  sway: (2.0 * Math.PI) / 180,
  drift: 0.010,

  layer({ motion, attached = "ground", clock, index = 0 }) {
    const t = clock, phase = index * 1.7;
    const wave = (period, offset = 0) => Math.sin((2 * Math.PI * t) / period + phase + offset);
    const still = { dx: 0, dy: 0, rotation: 0, wings: 1 };
    if (motion === "float" && attached === "ground") {
      const swell = 0.7 * wave(4.3) + 0.3 * Math.sin((2 * Math.PI * t) / 2.7);
      return { ...still, dy: this.bob * (0.5 + 0.5 * swell), rotation: this.bobRoll * wave(5.1, 0.8) };
    }
    if (motion === "float") return { ...still, dx: this.hoverX * wave(7.9), dy: this.hoverY * wave(5.3) };
    if (motion === "flutter") {
      const cycle = (t + phase) % this.wingCycle;
      const fold = cycle < this.wingFold ? Math.sin((Math.PI * cycle) / this.wingFold) ** 2 : 0;
      return { ...still, dx: this.hoverX * wave(7.9), dy: this.hoverY * wave(5.3),
        wings: 1 - (1 - this.wingsFolded) * fold };
    }
    if (motion === "sway") return { ...still, rotation: this.sway * (0.65 * wave(4.6) + 0.35 * wave(1.9)) };
    if (motion === "drift") return { ...still, dx: this.drift * wave(14) };
    return still;
  },

  /** How alive a world is `seconds` after it appeared: it comes alive over 3 s (scene.dart WorldScene.wake). */
  wake: (seconds) => {
    // Flutter's Curves.easeInOut: the cubic Bézier (0.42, 0) (0.58, 1), solved for x by bisection.
    const x = Math.min(Math.max(seconds / 3, 0), 1);
    const bezier = (t, p1, p2) => 3 * (1 - t) * (1 - t) * t * p1 + 3 * (1 - t) * t * t * p2 + t * t * t;
    let lo = 0, hi = 1, t = x;
    for (let i = 0; i < 30; i++) {
      t = (lo + hi) / 2;
      if (bezier(t, 0.42, 0.58) < x) lo = t; else hi = t;
    }
    return bezier(t, 0, 1);
  },

  /** A pose `k` of the way from still to full (motion.dart LayerPose.scaled). */
  scaled: (pose, k) => ({ dx: pose.dx * k, dy: pose.dy * k, rotation: pose.rotation * k, wings: 1 - (1 - pose.wings) * k }),

  /** A lantern's flicker (the same as the light it throws in living.frag). */
  flicker: (t) => Math.sin(2 * Math.PI * 1.7 * t) * 0.5 + Math.sin(2 * Math.PI * 2.9 * t + 1.1) * 0.3
    + Math.sin(2 * Math.PI * 5.3 * t + 2.3) * 0.2,
};

/** The painting's rect covering a box, cropped around `focus` (as on the phone). */
function coverRect(boxW, boxH, aspect, focus, highest = 0.62, maxZoom = 1.25) {
  let width = boxW / boxH > aspect ? boxW : boxH * aspect;
  let height = width / aspect;
  if (focus[1] > highest && focus[1] < 1) {
    const zoom = Math.min(Math.max((boxH * (1 - highest)) / (height * (1 - focus[1])), 1), maxZoom);
    width *= zoom;
    height *= zoom;
  }
  const targetY = boxH * (focus[1] > highest ? highest : focus[1]);
  const left = Math.min(Math.max(boxW / 2 - focus[0] * width, boxW - width), 0);
  const top = Math.min(Math.max(targetY - focus[1] * height, boxH - height), 0);
  return { left, top, width, height };
}

/** A seeded random sequence (the phone draws its particles from Dart's; the kind of drift is the same). */
function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** What drifts through the air (living.dart WorldParticles), in the painting's rect: width × height. */
function drawParticles(ctx, kind, clock, width, height) {
  const count = { leaves: 7, snow: 28, pollen: 18, fireflies: 12 }[kind] || 0;
  const leafColors = ["#D69630", "#E2AC40", "#BE6E28"];
  const t = clock;
  for (let i = 0; i < count; i++) {
    const r = seeded(7 + i * 31);
    const x0 = r(), start = r() * 40, s = r();
    ctx.save();
    if (kind === "leaves") {
      const speed = ((26 + 16 * s) * height) / 960;
      const age = (t + start) % ((height + 40) / speed);
      const x = x0 * width + ((18 * width) / 540) * Math.sin(age * 1.3 + start);
      const w = ((5 + 3 * s) * width) / 540;
      const tilt = Math.abs(Math.sin(age * (1.5 + 1.5 * s)));
      ctx.translate(x, -20 + age * speed);
      ctx.rotate(Math.sin(age * 0.9 + i) * 0.8);
      ctx.globalAlpha *= 0.82;
      ctx.fillStyle = leafColors[i % leafColors.length];
      ctx.beginPath();
      ctx.ellipse(0, 0, w, (w * (0.35 + 0.65 * tilt)) / 2, 0, 0, 2 * Math.PI);
      ctx.fill();
    } else if (kind === "snow") {
      const speed = ((18 + 22 * s) * height) / 960;
      const age = (t + start) % (height / speed + 1);
      ctx.globalAlpha *= 0.75;
      ctx.fillStyle = "#FFFFFF";
      ctx.beginPath();
      ctx.arc(x0 * width + 10 * Math.sin(age * 0.8 + start), age * speed - 6, ((1.2 + 1.8 * s) * width) / 540, 0, 2 * Math.PI);
      ctx.fill();
    } else if (kind === "pollen") {
      const x = (x0 + 0.04 * Math.sin(t / (5 + 3 * s) + i)) * width;
      const y = (0.35 + 0.45 * s + 0.03 * Math.sin(t / (4 + 2 * s) + i * 1.3)) * height;
      ctx.globalAlpha *= 0.55 * (0.6 + 0.4 * Math.sin(t * 0.7 + i));
      ctx.fillStyle = "#FFF4D6";
      ctx.beginPath();
      ctx.arc(x, y, (1.4 * width) / 540, 0, 2 * Math.PI);
      ctx.fill();
    } else if (kind === "fireflies") {
      const glow = Math.max(0, Math.sin(t * (0.8 + 0.6 * s) + i * 2.1));
      if (glow > 0) {
        const x = (x0 + 0.05 * Math.sin(t / (3 + 2 * s) + i)) * width;
        const y = (0.45 + 0.4 * s + 0.04 * Math.sin(t / (2.5 + s) + i * 0.7)) * height;
        const radius = (2.2 * width) / 540;
        const halo = ctx.createRadialGradient(x, y, 0, x, y, radius * 3 + 4);
        halo.addColorStop(0, `rgba(255, 226, 122, ${0.25 * glow})`);
        halo.addColorStop(1, "rgba(255, 226, 122, 0)");
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.arc(x, y, radius * 3 + 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = `rgba(255, 243, 176, ${0.9 * glow})`;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    ctx.restore();
  }
}

// The living painting and its layers, drawn as the phone draws them, with
// WebGL 2: the background through the phone's own shader (living.frag, the
// same file, its Flutter header swapped for WebGL's), each layer moved about
// its anchor as Flutter's Transform moves it. One context draws every card;
// each card copies the result onto its own canvas.
const LivingGL = (() => {
  let gl = null, canvas = null, failed = false, background = null, layer = null, empty = null;
  const textures = new WeakMap();

  const QUAD_VERTEX = `#version 300 es
in vec2 aCorner;
void main() { gl_Position = vec4(aCorner * 2.0 - 1.0, 0.0, 1.0); }`;

  // As Flutter's Transform(alignment: anchor): translate, rotate, then the wings' width, about the anchor.
  const LAYER_VERTEX = `#version 300 es
in vec2 aCorner;
uniform vec4 uRect;    // the painting's rect: left, top, width, height (px, y down)
uniform vec2 uAnchor;  // the pivot (px)
uniform vec2 uMove;    // dx, dy (px)
uniform vec3 uTurn;    // cos and sin of the rotation, the wings' share of their width
uniform vec2 uCanvas;  // the canvas (px)
out vec2 vUv;
void main() {
  vec2 p = uRect.xy + aCorner * uRect.zw - uAnchor;
  p.x *= uTurn.z;
  p = vec2(uTurn.x * p.x - uTurn.y * p.y, uTurn.y * p.x + uTurn.x * p.y);
  vec2 q = uAnchor + uMove + p;
  vUv = aCorner;
  gl_Position = vec4(q.x / uCanvas.x * 2.0 - 1.0, 1.0 - q.y / uCanvas.y * 2.0, 0.0, 1.0);
}`;
  // Premultiplied; a lantern's flicker scales its colour (Flutter's ColorFilter.matrix).
  const LAYER_FRAGMENT = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uLayer;
uniform float uBright;
out vec4 fragColor;
void main() {
  vec4 c = texture(uLayer, vUv);
  fragColor = vec4(min(c.rgb * uBright, vec3(c.a)), c.a);
}`;

  /** living.frag for WebGL 2: the same body; gl_FragCoord (y up) stands in for FlutterFragCoord (y down). */
  function forWebGL(source) {
    const swaps = [
      [/^#version 460 core$/m, "#version 300 es"],
      [/^#include <flutter\/runtime_effect\.glsl>$/m, ""],
      [/^uniform vec2 uSize;/m, "uniform float uCanvasH; // the canvas's height: gl_FragCoord's y is up\nuniform vec2 uSize;"],
      [/FlutterFragCoord\(\)\.xy/, "vec2(gl_FragCoord.x, uCanvasH - gl_FragCoord.y)"],
    ];
    for (const [pattern, replacement] of swaps) {
      if (!pattern.test(source)) throw new Error(`living.frag changed: ${pattern}`);
      source = source.replace(pattern, replacement);
    }
    return source;
  }

  function program(vertex, fragment) {
    const compiled = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const p = gl.createProgram();
    gl.attachShader(p, compiled(gl.VERTEX_SHADER, vertex));
    gl.attachShader(p, compiled(gl.FRAGMENT_SHADER, fragment));
    gl.bindAttribLocation(p, 0, "aCorner");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const at = {};
    const count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const name = gl.getActiveUniform(p, i).name;
      at[name] = gl.getUniformLocation(p, name);
    }
    return { p, at };
  }

  /** Starts once: true when WebGL 2 and the shader are ready. */
  async function start(shaderUrl) {
    if (gl || failed) return !failed;
    try {
      canvas = document.createElement("canvas");
      gl = canvas.getContext("webgl2", { alpha: true, premultipliedAlpha: true, antialias: false });
      if (!gl) throw new Error("no WebGL 2");
      const source = await (await fetch(shaderUrl)).text();
      background = program(QUAD_VERTEX, forWebGL(source));
      layer = program(LAYER_VERTEX, LAYER_FRAGMENT);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      empty = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, empty);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
      return true;
    } catch (error) {
      console.warn("ELYND Worlds: the living painting needs WebGL 2; showing the still pictures", error);
      failed = true;
      gl = null;
      return false;
    }
  }

  /** A picture as a texture (made once). The plate as the phone samples it (smooth, mipmapped). */
  function texture(image, { premultiply = false, mipmaps = false } = {}) {
    let t = textures.get(image);
    if (t) return t;
    t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, premultiply);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    if (mipmaps) gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mipmaps ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    textures.set(image, t);
    return t;
  }

  /** One card: width × height device pixels, the painting at `rect` (device pixels). Returns the canvas. */
  function draw(view, width, height, rect, clock, still, wake) {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    const living = view.world.living || {};
    const regions = view.regions;
    const mapped = Boolean(regions) && Boolean(living.water || living.sky || living.grass || living.stream);
    const on = (v) => (v && mapped ? 1 : 0);
    const light = (living.lights || [])[0];

    gl.disable(gl.BLEND);
    gl.useProgram(background.p);
    const u = background.at;
    gl.uniform1f(u.uCanvasH, height);
    gl.uniform2f(u.uSize, rect.width, rect.height);
    gl.uniform2f(u.uOrigin, rect.left, rect.top);
    gl.uniform1f(u.uTime, still ? 0 : clock % 3600);
    gl.uniform1f(u.uMotion, still ? 0 : wake);
    gl.uniform4f(u.uOn, on(living.water), on(living.sky), on(living.grass), on(living.stream));
    gl.uniform4f(u.uScene, living.trees ? 1 : 0, living.horizon ?? -1, 0, 0);
    gl.uniform4f(u.uLight, light ? light.x : 0, light ? light.y : 0, light ? light.size : 0, light ? 1 : 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture(view.background, { mipmaps: true }));
    gl.uniform1i(u.uPlate, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, mapped ? texture(regions) : empty);
    gl.uniform1i(u.uRegions, 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(layer.p);
    const l = layer.at;
    gl.uniform4f(l.uRect, rect.left, rect.top, rect.width, rect.height);
    gl.uniform2f(l.uCanvas, width, height);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(l.uLayer, 0);
    // Farthest first, nearest on top (layers are listed nearest first).
    for (let i = view.layers.length - 1; i >= 0; i--) {
      const spec = view.world.layers[i];
      const pose = still ? { dx: 0, dy: 0, rotation: 0, wings: 1 }
        : WorldMotion.scaled(WorldMotion.layer({ motion: spec.motion, attached: spec.attached, clock, index: i }), wake);
      gl.uniform2f(l.uAnchor, rect.left + spec.anchor[0] * rect.width, rect.top + spec.anchor[1] * rect.height);
      gl.uniform2f(l.uMove, pose.dx * rect.height, pose.dy * rect.height);
      gl.uniform3f(l.uTurn, Math.cos(pose.rotation), Math.sin(pose.rotation), pose.wings);
      gl.uniform1f(l.uBright, spec.glows && !still ? 1 + 0.12 * wake * WorldMotion.flicker(clock) : 1);
      gl.bindTexture(gl.TEXTURE_2D, texture(view.layers[i], { premultiply: true }));
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    return canvas;
  }

  return { start, draw, get ready() { return Boolean(gl); } };
})();

/** One world drawn on one canvas. */
class WorldView {
  constructor(canvas, world, base, shaderUrl = "living.frag") {
    this.canvas = canvas;
    this.world = world;
    this.ready = false;
    const url = (file) => `${base}/${world.id}/${file}`;
    const image = (file) =>
      new Promise((resolve, reject) => {
        const picture = new Image();
        picture.onload = () => resolve(picture);
        picture.onerror = reject;
        picture.src = url(file);
      });
    // The region map exactly as stored: its alpha is 1 - stream, and must not be premultiplied away.
    const exact = (file) => fetch(url(file))
      .then((response) => (response.ok ? response.blob() : Promise.reject(new Error(file))))
      .then((blob) => createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" }));
    const moving = Boolean(world.plate) && world.layers.length > 0;
    const regions = world.living && world.living.regions;
    Promise.all([
      image(moving ? world.plate : world.painting),
      Promise.all(moving ? world.layers.map((l) => image(l.file)) : []),
      regions ? exact(regions).catch(() => null) : null,
      LivingGL.start(shaderUrl),
    ])
      .then(([background, layers, map]) => {
        this.background = background;
        this.layers = layers;
        this.regions = map;
        this.bornAt = null; // set on the first frame drawn: it comes alive from there
        this.ready = true;
      })
      .catch(() => canvas.classList.add("failed"));
    // At rest: fully shown, as while the listener arrives.
    this.idle = WorldTimeline.build(600, [[0, 600]], [], false);
  }

  /** `session`: null at rest, else {timeline, t} of a played meditation. */
  draw(clock, session, still) {
    if (!this.ready) return;
    const canvas = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, pw, ph);
    const timeline = session ? session.timeline : this.idle;
    const t = session ? session.t : clock % 600;
    ctx.globalAlpha = session ? WorldTimeline.at(timeline.visibility, t) : 1;
    const rect = coverRect(pw, ph, this.world.aspect, this.world.focus);
    if (this.bornAt === null) this.bornAt = clock;
    const wake = still ? 0 : WorldMotion.wake(clock - this.bornAt);

    if (LivingGL.ready) {
      ctx.drawImage(LivingGL.draw(this, pw, ph, rect, clock, still, wake), 0, 0);
    } else {
      // Without WebGL 2: the pictures and their motion, no living painting.
      ctx.drawImage(this.background, rect.left, rect.top, rect.width, rect.height);
      for (let i = this.layers.length - 1; i >= 0; i--) {
        const spec = this.world.layers[i];
        const pose = still ? { dx: 0, dy: 0, rotation: 0, wings: 1 }
          : WorldMotion.scaled(WorldMotion.layer({ motion: spec.motion, attached: spec.attached, clock, index: i }), wake);
        const ax = rect.left + spec.anchor[0] * rect.width, ay = rect.top + spec.anchor[1] * rect.height;
        ctx.save();
        ctx.translate(ax + pose.dx * rect.height, ay + pose.dy * rect.height);
        ctx.rotate(pose.rotation);
        ctx.scale(pose.wings, 1);
        ctx.translate(-ax, -ay);
        ctx.drawImage(this.layers[i], rect.left, rect.top, rect.width, rect.height);
        ctx.restore();
      }
    }
    const particles = this.world.living && this.world.living.particles;
    if (particles && particles !== "none" && !still) {
      // In the phone's units (logical pixels), so they drift as there.
      ctx.save();
      ctx.globalAlpha *= wake;
      ctx.scale(dpr, dpr);
      ctx.translate(rect.left / dpr, rect.top / dpr);
      drawParticles(ctx, particles, clock, rect.width / dpr, rect.height / dpr);
      ctx.restore();
    }
  }
}
