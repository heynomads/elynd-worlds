// ELYND Worlds in the browser: the same timeline and motion as the phone
// (flutter/elynd_worlds_scene/lib/src/timeline.dart and motion.dart) and the
// engine (elynd_worlds/timeline.py). web/parity.html checks this timeline
// against the Python one, frame for frame.
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

// How a world moves: slowly, and only as such things move in nature (as
// flutter/elynd_worlds_scene/lib/src/motion.dart). Nothing grows or shrinks:
// still things stay still, a boat on the water bobs and rolls a little,
// something in the air loops slowly, plants sway in a light wind, mist drifts.
// The whole world drifts like a very slow camera, near layers a little more
// than the background (depth).
const WorldMotion = {
  cameraHeadroom: 1.03,
  cameraDrift: 0.008,
  bob: 0.0022,
  bobRoll: (0.4 * Math.PI) / 180,
  hover: 0.003,
  sway: (0.7 * Math.PI) / 180,
  drift: 0.006,
  parallax: (depth) => (depth === "near" ? 1.6 : 1.25),

  camera(clock) {
    return {
      dx: this.cameraDrift * Math.sin((2 * Math.PI * clock) / 47),
      dy: this.cameraDrift * 0.5 * Math.sin((2 * Math.PI * clock) / 61 + 1.3),
    };
  },

  layer({ motion, attached = "ground", clock, index }) {
    const phase = index * 1.7;
    const wave = (period, offset = 0) => Math.sin((2 * Math.PI * clock) / period + phase + offset);
    if (motion === "float" && attached === "ground") {
      return { dx: 0, dy: -this.bob * (0.75 * wave(8.7) + 0.25 * wave(4.9, 0.6)), rotation: this.bobRoll * wave(10.9, 1.1) };
    }
    if (motion === "float") return { dx: this.hover * wave(13.7), dy: this.hover * 0.8 * wave(9.1, 0.4), rotation: 0 };
    if (motion === "sway") return { dx: 0, dy: 0, rotation: this.sway * (0.65 * wave(6.8) + 0.35 * wave(3.7, 1.9)) };
    if (motion === "drift") return { dx: this.drift * wave(37), dy: 0, rotation: 0 };
    return { dx: 0, dy: 0, rotation: 0 };
  },
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

/** One world drawn on one canvas. */
class WorldView {
  constructor(canvas, world, base) {
    this.canvas = canvas;
    this.world = world;
    this.ready = false;
    const load = (src) =>
      new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = `${base}/${world.id}/${src}`;
      });
    const moving = world.plate && world.layers.length;
    Promise.all([load(moving ? world.plate : world.painting), ...(moving ? world.layers.map((l) => load(l.file)) : [])])
      .then(([background, ...layers]) => {
        this.background = background;
        this.layers = layers;
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
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const timeline = session ? session.timeline : this.idle;
    const t = session ? session.t : clock % 600;
    const visibility = session ? WorldTimeline.at(timeline.visibility, t) : 1;
    const rect = coverRect(w, h, this.world.aspect, this.world.focus);
    const camera = still ? { dx: 0, dy: 0 } : WorldMotion.camera(clock);

    ctx.save();
    ctx.globalAlpha = visibility;
    ctx.translate(w / 2, h / 2);
    ctx.scale(WorldMotion.cameraHeadroom, WorldMotion.cameraHeadroom);
    ctx.translate(-w / 2, -h / 2);
    ctx.drawImage(this.background, rect.left + camera.dx * rect.height, rect.top + camera.dy * rect.height,
      rect.width, rect.height);
    // Farthest first, nearest on top (layers are listed nearest first).
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const spec = this.world.layers[i];
      const pose = still ? { dx: 0, dy: 0, rotation: 0 }
        : WorldMotion.layer({ motion: spec.motion, attached: spec.attached, clock, index: i });
      const depth = WorldMotion.parallax(spec.depth);
      const ax = rect.left + spec.anchor[0] * rect.width;
      const ay = rect.top + spec.anchor[1] * rect.height;
      ctx.save();
      ctx.translate(ax + (camera.dx * depth + pose.dx) * rect.height, ay + (camera.dy * depth + pose.dy) * rect.height);
      ctx.rotate(pose.rotation);
      ctx.translate(-ax, -ay);
      ctx.drawImage(this.layers[i], rect.left, rect.top, rect.width, rect.height);
      ctx.restore();
    }
    ctx.restore();
  }
}
