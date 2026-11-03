// The X-ray of a world: how it was made, one step at a time, on the world's own canvas. 1 the painting as Muse
// Image painted it (with Muse Spark's design and the judges' scores), 2 SAM 3.1's layers and region map over it,
// 3 the plate and where it differs from the painting, 4 the living world again, drawn by the phone's own shader.
// Every step names its models and their times from the world's own record (web/worlds/index.json); a cost or a
// time that was not recorded says so. Only drawing with the 2D canvas over pictures the page already has: no
// library, nothing new to download but the painting itself when the world shows its plate.
"use strict";

const XRay = (() => {
  const STEPS = [
    { key: "painting", short: "Painting", title: "The painting" },
    { key: "sam", short: "SAM 3.1", title: "SAM 3.1: layers and regions" },
    { key: "plate", short: "Plate", title: "The plate: the scene without what moves" },
    { key: "live", short: "Live", title: "Live: the phone's shader" },
  ];
  // The region map stores water in red, sky in green, grass in blue and 1 - stream in alpha (living.frag);
  // here each gets a colour of its own.
  const REGIONS = [
    { name: "water", rgb: [86, 170, 255] },
    { name: "sky", rgb: [206, 176, 255] },
    { name: "grass", rgb: [132, 222, 120] },
    { name: "stream", rgb: [64, 230, 214] },
  ];
  // Painting and plate are separate WebP files: below this difference in a colour channel (of 255) it is the
  // files' compression, not the repaint (measured on the gallery's six worlds: the noise stays under 8).
  const CHANGED = 16;
  const FADE_MS = 220;

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const seconds = (s) => (typeof s === "number" && s > 0 ? `${s.toFixed(1)} s` : "time not recorded");
  const money = (v) => `$${v < 0.01 ? v.toFixed(4) : v.toFixed(3)}`;
  const judgeOf = (id) => (/open/.test(id || "") ? "Muse Spark Open" : "Muse Spark");
  const percent = (v) => `${v < 0.01 && v > 0 ? "<1" : Math.round(v * 100)} %`;
  const and = (items) => (items.length < 2 ? items.join("")
    : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`);
  const the = (names) => names.map((n) => `the ${n}`);

  function motionOf(layer) {
    if (layer.glows) return "glows";
    if (layer.motion === "float") return layer.attached === "air" ? "floats" : "bobs on the swell";
    return { flutter: "flutters", sway: "sways", drift: "drifts" }[layer.motion] || "stays still";
  }
  const SHORT_MOTION = (layer) => (layer.glows ? "glows" : layer.motion === "float" && layer.attached !== "air" ? "bobs"
    : layer.motion === "float" ? "floats" : { flutter: "flutters", sway: "sways", drift: "drifts" }[layer.motion] || "still");

  function image(src) {
    return new Promise((resolve, reject) => {
      const picture = new Image();
      picture.onload = () => resolve(picture);
      picture.onerror = () => reject(new Error(`could not load ${src}`));
      picture.src = src;
    });
  }

  function canvas2d(width, height) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return [canvas, canvas.getContext("2d", { willReadFrequently: true })];
  }

  /** The region map in colour, and how much of the picture each region covers (0-1), as the shader reads it. */
  function regionTint(map, living) {
    const [canvas, g] = canvas2d(map.width, map.height);
    g.drawImage(map, 0, 0);
    const d = g.getImageData(0, 0, map.width, map.height).data;
    const out = g.createImageData(map.width, map.height);
    const o = out.data;
    const on = REGIONS.map((r) => Boolean(living[r.name]));
    const sums = [0, 0, 0, 0];
    for (let i = 0; i < d.length; i += 4) {
      const values = [d[i] / 255, d[i + 1] / 255, d[i + 2] / 255, 1 - d[i + 3] / 255];
      let r = 0, gr = 0, b = 0, total = 0, top = 0;
      for (let k = 0; k < 4; k++) {
        const v = on[k] ? values[k] : 0;
        if (v <= 0) continue;
        sums[k] += v;
        total += v;
        top = Math.max(top, v);
        r += v * REGIONS[k].rgb[0];
        gr += v * REGIONS[k].rgb[1];
        b += v * REGIONS[k].rgb[2];
      }
      if (total < 0.02) continue;
      o[i] = r / total;
      o[i + 1] = gr / total;
      o[i + 2] = b / total;
      o[i + 3] = 150 * Math.min(top, 1);
    }
    g.putImageData(out, 0, 0);
    const area = map.width * map.height;
    return { canvas, coverage: Object.fromEntries(REGIONS.map((r, k) => [r.name, on[k] ? sums[k] / area : 0])) };
  }

  /** Where the plate differs from the painting (gold), and on how much of the picture (0-1). */
  function plateDiff(painting, plate) {
    const width = 360, height = Math.round((width * painting.naturalHeight) / painting.naturalWidth);
    const [canvas, a] = canvas2d(width, height);
    const [, b] = canvas2d(width, height);
    a.drawImage(painting, 0, 0, width, height);
    b.drawImage(plate, 0, 0, width, height);
    const pa = a.getImageData(0, 0, width, height).data;
    const pb = b.getImageData(0, 0, width, height).data;
    const out = a.createImageData(width, height);
    const o = out.data;
    let changed = 0;
    for (let i = 0; i < pa.length; i += 4) {
      const m = Math.max(Math.abs(pa[i] - pb[i]), Math.abs(pa[i + 1] - pb[i + 1]), Math.abs(pa[i + 2] - pb[i + 2]));
      if (m <= CHANGED) continue;
      changed++;
      o[i] = 255;
      o[i + 1] = 214;
      o[i + 2] = 140;
      o[i + 3] = Math.min(100, 40 + 2 * m); // a light veil: the repaint shows through
    }
    a.putImageData(out, 0, 0);
    return { canvas, changed: changed / (width * height) };
  }

  /** What one X-ray needs beyond what the world view already loaded. */
  async function load(view) {
    await view.loaded;
    if (!view.ready) throw new Error(`${view.world.id}: the world did not load`);
    const world = view.world;
    const url = (file) => `${view.base}/${world.id}/${file}`;
    const moving = Boolean(world.plate) && world.layers.length > 0;
    const painting = moving ? await image(url(world.painting)) : view.background;
    const plate = world.plate ? (moving ? view.background : await image(url(world.plate))) : null;
    return {
      painting,
      plate,
      tint: view.regions ? regionTint(view.regions, world.living || {}) : null,
      diff: plate ? plateDiff(painting, plate) : null,
    };
  }

  // ── What each step says, from the world's record ──────────────────────────────

  function rowsOf(world, key, assets) {
    const rows = [];
    const row = (who, what, step, cost) => rows.push({ who, what, time: step ? step.seconds : undefined, cost });
    if (key === "painting" && world.place) {
      const place = world.place;
      rows.push({ who: "Llama 3.3", what: place.rerun ? "described the place (re-run: production keeps no copy)"
        : "described the place", time: place.seconds, cost: place.usd });
    }
    for (const step of world.steps || []) {
      const base = step.step.replace(/ \d+$/, "");
      const n = Number((step.step.match(/ (\d+)$/) || [0, 1])[1]);
      const note = step.note || "";
      if (key === "painting") {
        if (base === "design") row("Muse Spark", "designed the world", step);
        else if (base === "paint") row("Muse Image", n > 1 ? "painted it again" : "painted it", step);
        else if (base === "critique") {
          const said = note.match(/^score (\d+)(?: by (\S+?))?:/);
          const who = said && said[2] ? judgeOf(said[2]) : judgeOf(world.verdict && world.verdict.judge);
          const score = said ? said[1] : world.verdict && world.verdict.score;
          row(who, score != null ? `judged it: ${score}/10` : "judged it", step, step.usd);
        } else if (base === "critique (open weights)") {
          const v = world.open_verdict;
          row("Muse Spark Open", v ? `judged it again: ${v.score}/10` : "judged it again", step);
        } else if (base === "revise") row("Muse Spark", "revised its design from the judge's note", step);
      } else if (key === "sam") {
        if (base === "cut") {
          const names = note.split(/,\s*/).filter(Boolean);
          if (world.layers.length) {
            const cut = world.layers.map((l) => `${esc(l.name)} (${SHORT_MOTION(l)})`);
            const count = world.layers.length === 1 ? "one layer" : `${world.layers.length} layers`;
            row("SAM 3.1", `cut <span class="sw box"></span>${count}: ${cut.join(", ")}`, step);
          } else if (names.length) {
            const these = and(the(names.map(esc)));
            const still = names.length === 1 ? "it stays still, so the world keeps it" : `${names.length === 2 ? "both" : "all"} stay still, so the world keeps them`;
            row("SAM 3.1", `cut out ${these}; ${still} as painted (no layer)`, step);
          } else row("SAM 3.1", "found nothing it could cut cleanly", step);
        } else if (base === "regions") {
          const names = note.split(/,\s*/).filter(Boolean);
          const shown = names.map((name) => {
            const region = REGIONS.find((r) => r.name === name);
            const share = assets && assets.tint && assets.tint.coverage[name];
            const swatch = region ? `<span class="sw" style="--c:rgb(${region.rgb})"></span>` : "";
            return `${swatch}${esc(name)}${share ? ` ${percent(share)}` : ""}`;
          });
          row("SAM 3.1", names.length ? `mapped ${shown.join(", ")}${assets && assets.tint ? " of the picture" : ""}`
            : "found nothing that moves in nature", step);
        }
      } else if (key === "plate") {
        if (base === "remove subjects") row("Muse Image", "repainted the scene without what moves", step);
        else if (base === "plate") {
          if (note === "painting") row("Plate", "none needed: nothing moves away", step);
          else {
            const what = {
              muse_image: "checked the repaint: kept only under each moving subject",
              filled: "found a subject left in the repaint, so it was filled in locally",
            }[note] || esc(note);
            row("SAM 3.1", what, step);
          }
        } else if (base === "replate") {
          // A plate rebuilt after the run with the current rule: the record's note, a long one shortened to what
          // was done and what it cost (the whole note is in the card's own list of steps).
          const costs = note.match(/\$\d+(?:\.\d+)?/g) || [];
          let what = note.replace(/^muse_image\b:?/, "Muse Image's repaint").replace(/^none:/, "no plate:")
            .replace(/^painting\b/, "the painting itself");
          if (what.length > 90) what = `Muse Image's repaint${/SAM 3\.1/.test(note) ? ", checked by SAM 3.1" : ""}`;
          rows.push({ who: "Rebuilt later:", what: esc(what), time: step.seconds, costText: costs.join(" + ") });
        }
      }
    }
    return rows;
  }

  function livelyOf(world) {
    const living = world.living || {};
    const parts = [];
    if (living.water) parts.push("the water ripples and glints");
    if (living.fish) parts.push("now and then a fish rises");
    if (living.stream) parts.push("the stream flows");
    if (living.grass) parts.push(living.shadows ? "the grass waves under drifting cloud shadows" : "the grass waves");
    if (living.trees) parts.push("the crowns sway");
    if (living.sky) parts.push(living.night ? "the clouds drift and the stars twinkle" : "the clouds drift");
    if (living.sky && !living.night && living.birds === "flock") parts.push("a flock crosses now and then");
    if (living.sky && !living.night && living.birds === "gulls") parts.push("gulls glide over the sea");
    if ((living.lights || []).length) parts.push("the lantern's light flickers");
    const drift = { leaves: "leaves", snow: "snow", pollen: "pollen", fireflies: "fireflies", rain: "rain", petals: "petals" }[living.particles];
    if (drift) parts.push(`${drift} drift${drift === "rain" ? "s" : ""} through the air`);
    for (const layer of world.layers || []) {
      if (layer.motion !== "still" && !layer.glows) parts.push(`the ${esc(layer.name)} ${motionOf(layer)}`);
    }
    return parts;
  }

  function totalOf(world) {
    if (typeof world.cost_usd !== "number") return "whole world not recorded";
    if (typeof world.meta_usd === "number" && typeof world.open_usd === "number") {
      return `whole world ${money(world.cost_usd)} (${money(world.meta_usd)} Meta's Model API + ${money(world.open_usd)} Muse Spark Open)`;
    }
    return `whole world ${money(world.cost_usd)} at Meta's prices`;
  }

  function caption(view, xr) {
    const world = view.world;
    const step = STEPS[xr.step];
    const assets = xr.assets;
    const head = `<p class="x-head"><span class="x-count">X-ray ${xr.step + 1}/${STEPS.length}</span> ${step.title}</p>`;
    let rows, extra = "", cost;
    if (step.key === "live") {
      const gl = LivingGL.ready;
      rows = [{ who: "living.frag", what: gl ? "the phone's own shader, here in WebGL 2"
        : "needs WebGL 2: here only the pictures move",
        time: "every frame, on the device" }];
      const parts = livelyOf(world);
      if (parts.length) extra += `<p class="x-note">${parts.join(", ").replace(/^./, (c) => c.toUpperCase())}.</p>`;
      const ready = typeof world.seconds_to_painting === "number"
        ? `Painted ${Math.round(world.seconds_to_painting)} s and alive ${Math.round(world.seconds)} s after the agent started`
        : typeof world.seconds === "number" ? `Moving ${Math.round(world.seconds)} s after the agent started` : "";
      // Which of the painting's checks and cuts this run had: they work at the same time.
      const had = (name) => (world.steps || []).some((s) => s.step.replace(/ \d+$/, "") === name);
      const together = [had("critique") && "the judge", (had("cut") || had("regions")) && "SAM",
        had("remove subjects") && "the repaint"].filter(Boolean);
      const overlap = together.length > 1 ? `; ${and(together)} work at the same time, so step times overlap` : "";
      if (ready) extra += `<p class="x-note">${ready}${overlap}.</p>`;
      cost = `Cost: $0, the shader calls no model · ${totalOf(world)}`;
    } else {
      rows = rowsOf(world, step.key, assets);
      if (xr.failed) extra += `<p class="x-note">The pictures for this step could not be loaded; the words are the world's record.</p>`;
      if (step.key === "plate") {
        let said;
        if (!world.plate) said = "No plate: the world is the painting itself.";
        else if (xr.failed) said = "";
        else if (!assets) said = "Comparing the plate with the painting…";
        else if (assets.diff.changed === 0) said = "The plate is the painting: no pixel differs.";
        else said = `<span class="sw gold"></span>Differs from the painting on ${percent(assets.diff.changed)} of the picture, where something moves away. Hold the picture to compare.`;
        if (said) extra += `<p class="x-note">${said}</p>`;
      }
      const recorded = rows.some((r) => typeof r.cost === "number" || r.costText);
      cost = `Cost: ${recorded ? "not recorded per step unless shown" : "not recorded per step"} · ${totalOf(world)}`;
    }
    const items = rows.map((r) => {
      const time = typeof r.time === "string" ? r.time : seconds(r.time);
      const price = typeof r.cost === "number" ? ` · ${money(r.cost)}` : r.costText ? ` · ${r.costText}` : "";
      return `<li><b>${esc(r.who)}</b> ${r.what} <span class="t">· ${time}${price}</span></li>`;
    }).join("");
    return `${head}${items ? `<ul class="x-rows">${items}</ul>` : ""}${extra}<p class="x-cost">${cost}</p>`;
  }

  // ── Drawing ───────────────────────────────────────────────────────────

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function label(ctx, text, x, y, dpr, maxX) {
    ctx.font = `600 ${12 * dpr}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
    const w = ctx.measureText(text).width + 14 * dpr, h = 20 * dpr;
    const left = Math.max(4 * dpr, Math.min(x, maxX - w - 4 * dpr));
    const top = Math.max(4 * dpr, y - h - 4 * dpr);
    ctx.fillStyle = "rgba(15, 36, 38, 0.82)";
    roundRect(ctx, left, top, w, h, 6 * dpr);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, left + 7 * dpr, top + h / 2 + 0.5 * dpr);
  }

  function boxes(ctx, layers, rect, dpr, width, named) {
    ctx.save();
    ctx.lineWidth = 1.5 * dpr;
    ctx.setLineDash([5 * dpr, 4 * dpr]);
    ctx.strokeStyle = "rgba(232, 210, 162, 0.95)";
    for (const layer of layers) {
      const [x0, y0, x1, y1] = layer.box;
      const x = rect.left + x0 * rect.width, y = rect.top + y0 * rect.height;
      roundRect(ctx, x, y, (x1 - x0) * rect.width, (y1 - y0) * rect.height, 6 * dpr);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    if (named) {
      for (const layer of layers) {
        label(ctx, layer.name, rect.left + layer.box[0] * rect.width, rect.top + layer.box[1] * rect.height, dpr, width);
      }
    }
    ctx.restore();
  }

  /** One still step, drawn into the X-ray's own cache. */
  function compose(view, xr, pw, ph, dpr) {
    if (!xr.cache) xr.cache = document.createElement("canvas");
    const cache = xr.cache;
    if (cache.width !== pw || cache.height !== ph) {
      cache.width = pw;
      cache.height = ph;
    }
    const ctx = cache.getContext("2d");
    const world = view.world, a = xr.assets;
    const rect = coverRect(pw, ph, world.aspect, world.focus);
    const put = (picture, alpha = 1) => {
      ctx.globalAlpha = alpha;
      ctx.drawImage(picture, rect.left, rect.top, rect.width, rect.height);
      ctx.globalAlpha = 1;
    };
    const shade = (alpha) => {
      ctx.fillStyle = `rgba(8, 18, 20, ${alpha})`;
      ctx.fillRect(0, 0, pw, ph);
    };
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, pw, ph);
    const key = STEPS[xr.step].key;
    if (key === "painting") {
      put(a.painting);
    } else if (key === "sam") {
      put(a.painting);
      shade(0.5);
      if (a.tint) put(a.tint.canvas);
      for (let i = view.layers.length - 1; i >= 0; i--) put(view.layers[i]); // the cut-outs, at full light
      boxes(ctx, world.layers, rect, dpr, pw, true);
    } else if (key === "plate") {
      if (!a.plate || xr.holding) {
        put(a.painting);
      } else {
        put(a.plate);
        shade(0.45);
        if (a.diff) put(a.diff.canvas);
        boxes(ctx, world.layers.filter((l) => l.motion !== "still" || l.glows), rect, dpr, pw, false);
      }
      // Level with the X-ray button (10 px from the top, 36 px tall): which picture this is.
      if (a.plate) label(ctx, xr.holding ? "The painting" : "The plate", 12 * dpr, 42 * dpr, dpr, pw);
    }
  }

  // ── The controls ───────────────────────────────────────────────────────

  /** Gives `view` (on `frame`, the box around its canvas) an X-ray toggle and steps; `panel` shows the words. */
  function attach(view, { card, frame, panel, title }) {
    const xr = { on: false, step: 0, dirty: true, assets: null, loading: null, holding: false, fade: null, snap: null,
      cache: null, size: "", label: view.canvas.getAttribute("aria-label") || title };
    view.xray = xr;
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "xray-toggle";
    toggle.setAttribute("aria-pressed", "false");
    toggle.setAttribute("aria-label", `X-ray: how ${title} was made`);
    // A layers glyph beside the word, so the X-ray reads as a control on the picture.
    toggle.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.4"'
      + ' stroke-linejoin="round"><path d="M8 1.8 14.2 5 8 8.2 1.8 5Z"/><path d="M1.8 8 8 11.2 14.2 8"/>'
      + '<path d="M1.8 11 8 14.2 14.2 11"/></svg>X-ray';
    const nav = document.createElement("div");
    nav.className = "xray-nav";
    nav.setAttribute("role", "group");
    nav.setAttribute("aria-label", `X-ray steps: ${title}`);
    nav.hidden = true;
    nav.innerHTML = STEPS.map((s, i) => `<button type="button" data-step="${i}"><span class="n">${i + 1}</span> ${s.short}</button>`).join("");
    frame.append(toggle, nav);
    const buttons = [...nav.querySelectorAll("button")];

    const snapshot = () => {
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || !view.canvas.width) return;
      if (!xr.snap) xr.snap = document.createElement("canvas");
      xr.snap.width = view.canvas.width;
      xr.snap.height = view.canvas.height;
      xr.snap.getContext("2d").drawImage(view.canvas, 0, 0);
      xr.fade = performance.now();
    };
    const render = () => {
      buttons.forEach((b, i) => (i === xr.step ? b.setAttribute("aria-current", "step") : b.removeAttribute("aria-current")));
      panel.innerHTML = caption(view, xr);
      panel.scrollTop = 0;
      // Words longer than the card's own: they scroll, the edge fades, and the keyboard can reach them.
      const more = xr.on && panel.scrollHeight > panel.clientHeight + 1;
      panel.classList.toggle("more", more);
      if (more) panel.tabIndex = 0;
      else panel.removeAttribute("tabindex");
      view.canvas.setAttribute("aria-label", xr.on
        ? `${title.charAt(0).toUpperCase()}${title.slice(1)}, X-ray step ${xr.step + 1} of ${STEPS.length}: ${STEPS[xr.step].title}`
        : xr.label);
    };
    const go = (step) => {
      const next = Math.max(0, Math.min(STEPS.length - 1, step));
      if (next === xr.step) return;
      snapshot();
      xr.step = next;
      xr.holding = false;
      xr.dirty = true;
      render();
    };
    const open = (on) => {
      snapshot();
      xr.on = on;
      xr.step = 0;
      xr.holding = false;
      xr.dirty = true;
      toggle.setAttribute("aria-pressed", String(on));
      card.classList.toggle("xray-on", on);
      nav.hidden = !on;
      if (on && !xr.loading) {
        xr.loading = load(view).then((assets) => {
          xr.assets = assets;
          xr.dirty = true;
          if (xr.on) render();
        }).catch((error) => {
          console.warn("ELYND Worlds X-ray:", error.message);
          xr.failed = true;
          if (xr.on) render();
        });
      }
      render();
    };
    // The card keeps one height whatever the X-ray shows: its words get the room of the tallest step, measured
    // with the longest numbers they can have, again whenever the card's width changes. Nothing moves when the
    // X-ray opens, steps or closes.
    const longest = { tint: { coverage: { water: 0.99, sky: 0.99, grass: 0.99, stream: 0.99 } }, diff: { changed: 0.99 } };
    const fit = () => {
      const swap = panel.parentElement;
      const probe = document.createElement("div");
      probe.className = "xray-panel x-probe";
      probe.setAttribute("aria-hidden", "true");
      swap.appendChild(probe);
      let tallest = 0;
      for (let i = 0; i < STEPS.length; i++) {
        probe.innerHTML = caption(view, { ...xr, step: i, assets: longest });
        tallest = Math.max(tallest, probe.getBoundingClientRect().height);
      }
      probe.remove();
      swap.style.minHeight = `${Math.ceil(tallest)}px`;
    };
    fit();
    // Watched on the picture's box, whose size never follows the words (so the new height cannot loop back).
    let width = frame.clientWidth;
    new ResizeObserver(() => {
      if (frame.clientWidth === width) return;
      width = frame.clientWidth;
      requestAnimationFrame(fit);
    }).observe(frame);

    toggle.addEventListener("click", () => open(!xr.on));
    buttons.forEach((b, i) => b.addEventListener("click", () => go(i)));
    nav.addEventListener("keydown", (event) => {
      const move = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!move) return;
      event.preventDefault();
      go(xr.step + move);
      buttons[xr.step].focus();
    });

    // On the picture: swipe sideways for the next or the previous step; on the plate, press and hold to see
    // the painting. Up and down still scroll the page (touch-action: pan-y in the page's style).
    let press = null;
    const stop = () => {
      if (press) clearTimeout(press.timer);
      if (xr.holding) {
        xr.holding = false;
        xr.dirty = true;
      }
      press = null;
    };
    frame.addEventListener("pointerdown", (event) => {
      if (!xr.on || event.target.closest("button") || (event.pointerType === "mouse" && event.button !== 0)) return;
      try { frame.setPointerCapture(event.pointerId); } catch (error) { /* a swipe that leaves the picture ends there */ }
      press = { x: event.clientX, y: event.clientY, timer: setTimeout(() => {
        if (press && STEPS[xr.step].key === "plate" && xr.assets && xr.assets.plate) {
          xr.holding = true;
          xr.dirty = true;
        }
      }, 150) };
    });
    frame.addEventListener("pointermove", (event) => {
      if (press && Math.hypot(event.clientX - press.x, event.clientY - press.y) > 12) clearTimeout(press.timer);
    });
    frame.addEventListener("pointerup", (event) => {
      if (!press) return;
      const dx = event.clientX - press.x, dy = event.clientY - press.y;
      const swiped = !xr.holding && Math.abs(dx) > 40 && Math.abs(dx) > 1.5 * Math.abs(dy);
      stop();
      if (swiped) go(xr.step + (dx < 0 ? 1 : -1));
    });
    frame.addEventListener("pointercancel", stop);
    frame.addEventListener("pointerleave", stop);
    frame.addEventListener("contextmenu", (event) => {
      if (xr.on) event.preventDefault();
    });
  }

  /** Draws `view` for this frame while its X-ray is open. */
  function draw(view, clock, session, still, now) {
    const xr = view.xray;
    // Closed (fading back), on the live step, or still loading its pictures: the living world.
    const live = !xr.on || STEPS[xr.step].key === "live" || !xr.assets;
    const canvas = view.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pw = Math.round(canvas.clientWidth * dpr), ph = Math.round(canvas.clientHeight * dpr);
    const fading = xr.fade !== null && now - xr.fade < FADE_MS;
    if (live) {
      view.draw(clock, session, still);
    } else {
      const size = `${pw}x${ph}`;
      if (xr.dirty || xr.size !== size) {
        compose(view, xr, pw, ph, dpr);
        xr.size = size;
      }
      if (!xr.dirty && !fading && canvas.width === pw && canvas.height === ph && !xr.wasFading) return;
      if (canvas.width !== pw || canvas.height !== ph) {
        canvas.width = pw;
        canvas.height = ph;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1;
      ctx.clearRect(0, 0, pw, ph);
      ctx.drawImage(xr.cache, 0, 0);
      xr.dirty = false;
    }
    xr.wasFading = fading;
    if (fading && xr.snap && xr.snap.width === canvas.width && xr.snap.height === canvas.height) {
      const ctx = canvas.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1 - (now - xr.fade) / FADE_MS;
      ctx.drawImage(xr.snap, 0, 0);
      ctx.globalAlpha = 1;
    } else if (!fading) {
      xr.fade = null;
    }
  }

  /** Whether `view` draws through its X-ray this frame (open, or still fading back to the living world). */
  const active = (view) => Boolean(view.xray) && (view.xray.on || view.xray.fade !== null);

  return { attach, draw, active, STEPS };
})();
