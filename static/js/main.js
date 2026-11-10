/* RoboScript project page: scroll reveal, the explain-on-hover stage chips, the
   main results table and figure tabs, the real-robot cards and the rollout wall.
   Numbers are copied from the paper's tables. Visual effects (theme, motion, lightbox,
   count-up) live in fx.js. */

/* ---------------- paper data ---------------- */

const CATS = [
    ["art", "Articulated"],
    ["ori", "Oriented parts"],
    ["fun", "Functional points"],
    ["rel", "Relative placement"],
    ["one", "Single objects"],
];
// task, category, pi0.5 C/R, Flash C/R, Luna C/R, Sol C/R  (appendix per-task table);
// the wall's SR badges read Sol's pair, and tools/build_promo.py reads this list too
const TASKS = [
    ["open_laptop", "art", 93, 68, 60, 60, 90, 100, 90, 100],
    ["open_microwave", "art", 86, 59, 100, 100, 100, 100, 100, 100],
    ["put_object_cabinet", "art", 39, 22, 60, 40, 30, 10, 90, 80],
    ["turn_switch", "art", 52, 25, 100, 100, 90, 90, 100, 90],
    ["adjust_bottle", "ori", 99, 77, 80, 90, 70, 70, 70, 80],
    ["beat_block_hammer", "ori", 93, 23, 90, 80, 90, 80, 90, 90],
    ["hanging_mug", "ori", 17, 14, 0, 0, 30, 0, 30, 20],
    ["move_stapler_pad", "ori", 20, 18, 100, 90, 70, 90, 90, 100],
    ["place_dual_shoes", "ori", 70, 45, 60, 80, 70, 60, 80, 80],
    ["place_fan", "ori", 66, 26, 10, 20, 30, 30, 40, 70],
    ["place_mouse_pad", "ori", 34, 31, 100, 100, 100, 100, 100, 90],
    ["place_phone_stand", "ori", 65, 37, 90, 100, 70, 80, 80, 90],
    ["place_shoe", "ori", 82, 50, 90, 80, 90, 90, 80, 80],
    ["rotate_qrcode", "ori", 90, 21, 70, 30, 70, 60, 30, 60],
    ["scan_object", "ori", 49, 40, 0, 30, 30, 20, 60, 60],
    ["place_bread_basket", "fun", 63, 60, 90, 80, 80, 70, 100, 100],
    ["place_bread_skillet", "fun", 76, 47, 30, 50, 20, 20, 60, 60],
    ["place_burger_fries", "fun", 82, 86, 90, 80, 100, 100, 100, 90],
    ["place_can_basket", "fun", 56, 10, 60, 80, 70, 70, 70, 40],
    ["place_cans_plasticbox", "fun", 31, 73, 90, 80, 100, 80, 100, 100],
    ["place_container_plate", "fun", 92, 72, 90, 90, 100, 60, 90, 90],
    ["place_empty_cup", "fun", 96, 80, 90, 80, 100, 90, 100, 90],
    ["place_object_basket", "fun", 70, 29, 80, 90, 60, 80, 90, 80],
    ["place_object_scale", "fun", 78, 38, 80, 90, 100, 90, 100, 100],
    ["place_object_stand", "fun", 85, 47, 90, 90, 80, 100, 90, 80],
    ["handover_block", "rel", 30, 12, 70, 60, 90, 80, 70, 50],
    ["move_can_pot", "rel", 60, 10, 80, 60, 50, 40, 50, 20],
    ["move_pillbottle_pad", "rel", 46, 32, 90, 90, 100, 80, 90, 100],
    ["place_a2b_left", "rel", 71, 49, 100, 90, 100, 90, 100, 100],
    ["place_a2b_right", "rel", 68, 39, 100, 100, 100, 90, 100, 100],
    ["stack_blocks_three", "rel", 77, 25, 60, 80, 100, 100, 100, 100],
    ["stack_blocks_two", "rel", 91, 43, 100, 100, 100, 100, 100, 100],
    ["stack_bowls_three", "rel", 81, 52, 40, 70, 20, 20, 30, 50],
    ["stack_bowls_two", "rel", 94, 70, 90, 70, 90, 80, 90, 90],
    ["stamp_seal", "rel", 53, 21, 70, 90, 90, 80, 100, 100],
    ["blocks_ranking_rgb", "one", 72, 44, 80, 70, 90, 80, 100, 100],
    ["blocks_ranking_size", "one", 45, 21, 70, 100, 100, 100, 100, 100],
    ["click_alarmclock", "one", 65, 50, 100, 90, 90, 90, 100, 100],
    ["click_bell", "one", 31, 35, 100, 100, 100, 100, 100, 100],
    ["dump_bin_bigbin", "one", 92, 84, 70, 60, 70, 60, 70, 70],
    ["grab_roller", "one", 99, 97, 100, 100, 90, 100, 100, 100],
    ["handover_mic", "one", 98, 6, 90, 100, 100, 100, 100, 100],
    ["lift_pot", "one", 98, 35, 100, 90, 100, 100, 100, 90],
    ["move_playingcard_away", "one", 84, 65, 90, 100, 100, 100, 100, 100],
    ["pick_diverse_bottles", "one", 66, 42, 30, 70, 30, 50, 50, 30],
    ["pick_dual_bottles", "one", 72, 57, 90, 100, 100, 100, 40, 60],
    ["press_stapler", "one", 79, 63, 100, 100, 90, 90, 100, 100],
    ["put_bottles_dustbin", "one", 78, 50, 70, 30, 50, 40, 60, 80],
    ["shake_bottle", "one", 100, 100, 100, 90, 100, 100, 100, 100],
    ["shake_bottle_horizontally", "one", 99, 100, 100, 100, 100, 90, 100, 100],
];
const TASK_BY_NAME = Object.fromEntries(TASKS.map(t => [t[0], t]));

// main table: name, demos, policy, agent, clean, rand, avg, retention
const MAIN = [
    { block: "Trained action policies" },
    { group: "Vision-language-action models (VLAs)" },
    ["StarVLA", "50", 1, 0, 46.5, 3.2, 24.8, 6.8],
    ["GR00T-N1.7", "50", 1, 0, 43.6, 20.7, 32.2, 47.5],
    ["GalaxeaVLA", "50", 1, 0, 62.7, 12.7, 37.7, 20.3],
    ["Xiaomi-Robotics-0", "50", 1, 0, 62.9, 18.2, 40.6, 28.9],
    ["Abot-M0", "50", 1, 0, 57.4, 30.4, 43.9, 52.9],
    ["X-VLA", "50", 1, 0, 68.0, 20.9, 44.5, 30.7],
    ["Spatial Forcing", "50", 1, 0, 77.2, 26.7, 52.0, 34.6],
    ["π<sub>0.5</sub>", "50", 1, 0, 70.7, 46.0, 58.4, 65.1],
    { group: "World action models (WAMs)" },
    ["AHA-WAM", "50", 1, 0, 64.3, 3.2, 33.8, 5.0],
    ["FastWAM", "50", 1, 0, 77.8, 1.9, 39.9, 2.4],
    ["X-WAM", "50", 1, 0, 70.0, 25.8, 47.9, 36.9],
    ["LingBot-VA", "50", 1, 0, 80.7, 34.6, 57.7, 42.9],
    ["4D-WAM", "50", 1, 0, 81.5, 41.8, 61.7, 51.3],
    { block: "Agents with trained policies" },
    ["HarnessVLA (frozen base)<sup>†</sup>", "<i>N</i>", 1, 0, null, 50.4, null, null],
    ["HarnessVLA (Codex)<sup>†</sup>", "<i>N</i> + 1", 1, 1, null, 58.0, null, null],
    ["HarnessVLA (Claude Code)<sup>†</sup>", "<i>N</i> + 1", 1, 1, null, 58.4, null, null],
    { block: "Action-policy-free agents" },
    ["RoboScript (DeepSeek-V4.1-Flash)", "1", 0, 1, 77.8, 78.4, 78.1, 100.8, true],
    ["RoboScript (GPT-5.6-Luna)", "1", 0, 1, 79.8, 76.0, 77.9, 95.2, true],
    ["RoboScript (GPT-6.1-Sol)", "1", 0, 1, 83.6, 83.2, 83.4, 99.5, true],
];

// real robot (dual-arm PiPER, memories rehearsed on a simulated ALOHA): manifest key,
// name, instruction, successes out of 10, the three clips the Real Robot section shows
// and the four the rollout wall shows (file name starts, chosen for different object
// placements), in the order of the paper's figure
const REAL = [
    ["click_bell", "Click Bell", "Click the bell's top center on the table.", 10, ["9f681320", "a3361294", "a53bf471"],
        ["08924bd7", "9f681320", "a3361294", "a53bf471"]],
    ["grab_roller", "Grab Roller", "Use both arms to grab the roller on the table.", 7, ["1814d1c1", "5a0fdb34", "97187405"],
        ["1814d1c1", "2866fb13", "6b081156", "c4030ee3"]],
    ["move_playingcard_away", "Move Playingcard Away", "Pick up the playing card and move it further away.", 7,
        ["11111b95", "6092289c", "a39bf7d0"], ["11111b95", "6092289c", "7a7066b5", "8a71d700"]],
    ["stack_bowls_two", "Stack Bowls Two", "Stack the two bowls on top of each other.", 6, ["IMG_0395", "IMG_0413", "IMG_0415"],
        ["IMG_0395", "IMG_0413", "IMG_0414", "IMG_0415"]],
];
const REAL_BY_KEY = Object.fromEntries(REAL.map(r => [r[0], r]));

/* ---------------- helpers ---------------- */

const $ = id => document.getElementById(id);
const fmt = v => v == null ? "&ndash;" : v.toFixed(1);
// the Motion switch in the nav (fx.js); it starts from prefers-reduced-motion
const calm = () => document.documentElement.dataset.motion === "off";

function renderMath(el) {
    if (window.renderMathInElement) {
        renderMathInElement(el, {
            delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
            throwOnError: false,
        });
    }
}

/* ---------------- scroll reveal ---------------- */

const riseIO = new IntersectionObserver((entries, obs) => {
    for (const e of entries) {
        if (!e.isIntersecting) continue;
        e.target.classList.add("is-in");
        obs.unobserve(e.target);
    }
}, { rootMargin: "0px 0px -10% 0px", threshold: 0.06 });

/* ---------------- chips that explain themselves ----------------
   Hover previews, click pins; the panel keeps a minimum height so previewing
   never moves the page under the cursor. */

function explainer(buttons, out, { flash = false } = {}) {
    const idle = out.innerHTML;
    let pinned = null;
    const paint = btn => {
        buttons.forEach(b => b.classList.toggle("is-on", b === btn));
        out.innerHTML = btn ? btn.dataset.desc : idle;
        renderMath(out);
    };
    const pin = btn => {
        pinned = pinned === btn ? null : btn;
        buttons.forEach(b => b.setAttribute("aria-pressed", String(b === pinned)));
        paint(pinned);
    };
    buttons.forEach(b => {
        b.addEventListener("click", () => pin(b));
        b.addEventListener("mouseenter", () => paint(b));
        b.addEventListener("focus", () => paint(b));
        b.addEventListener("mouseleave", () => paint(pinned));
        b.addEventListener("blur", () => paint(pinned));
    });
    if (flash && !calm()) {
        const io = new IntersectionObserver((es, obs) => {
            if (!es.some(e => e.isIntersecting)) return;
            obs.disconnect();
            buttons.forEach((b, i) => setTimeout(() => {
                b.classList.add("flash");
                b.addEventListener("animationend", () => b.classList.remove("flash"), { once: true });
            }, i * 90));
        }, { threshold: .5 });
        io.observe(buttons[0].parentElement);
    }
}

/* ---------------- main results table ---------------- */

function buildMainTable() {
    const body = document.querySelector("#mainTable tbody");
    const rows = MAIN.filter(Array.isArray);
    // best and second best per success column
    const rank = {};
    for (const col of [4, 5, 6]) {
        const vals = [...new Set(rows.map(r => r[col]).filter(v => v != null))].sort((a, b) => b - a);
        rank[col] = [vals[0], vals[1]];
    }
    let i = 0;
    const cell = (v, col) => {
        if (v == null) return `<td class="num muted">&ndash;</td>`;
        const cls = v === rank[col][0] ? " best" : v === rank[col][1] ? " second" : "";
        const d = Math.min(i * 28, 700);
        return `<td class="num"><span class="vbar" style="--v:${Math.min(v, 100)};--d:${d}ms"><span class="${cls.trim()}">${fmt(v)}</span><i></i></span></td>`;
    };
    body.innerHTML = MAIN.map(r => {
        if (r.block) return `<tr class="block"><td colspan="8">${r.block}</td></tr>`;
        if (r.group) return `<tr class="group"><td colspan="8">${r.group}</td></tr>`;
        i++;
        const [name, demos, pol, agt, c, rr, avg, ret, ours] = r;
        const mark = on => on ? `<span class="yes" aria-label="yes">&check;</span>` : `<span class="no" aria-label="no">&times;</span>`;
        return `<tr class="${ours ? "ours" : ""}">
            <td class="name">${name}</td>
            <td class="ctr">${demos}</td>
            <td class="ctr">${mark(pol)}</td>
            <td class="ctr">${mark(agt)}</td>
            ${cell(c, 4)}${cell(rr, 5)}${cell(avg, 6)}
            <td class="num">${fmt(ret)}</td>
        </tr>`;
    }).join("");
    riseIO.observe($("mainTableWrap"));
}

/* ---------------- rollouts ----------------
   RoboTwin 2.0 first, one tile per task with its PER_TASK episodes stacked like
   cards (a click deals the next), then the real robot, one tile per episode (a
   click opens it large). RoboDojo is in the manifest too, but only scrolls past in
   the marquee under the hero. Posters load as tiles come near, and clips play only
   while their tile is on screen. */

const M = window.MANIFEST || {};
const posterOf = src => src.replace(/^(video|static\/videos)\//, "static/posters/").replace(/\.mp4$/, ".jpg");
const pretty = name => REAL_BY_KEY[name] ? REAL_BY_KEY[name][1] : name.replace(/_/g, " ");

// The real-robot shelf has at least REAL_TASKS tasks. Clips go under
// video/Real/<Task Name>/ (then run tools/build_assets.py). A task whose folder is
// still empty, or one not named yet, shows PH_TILES blurred placeholder tiles that
// borrow recorded footage.
const REAL_TASKS = 4;
const PH_TILES = 3;

const BENCH = { real: "Real robot", robotwin: "RoboTwin 2.0", robodojo: "RoboDojo" };

function collectEpisodes() {
    const eps = { real: [], robotwin: [], robodojo: [] };
    // the paper's tasks in its order, then any other folder under video/Real/
    const realKeys = Object.keys(M.real || {}).sort((a, b) =>
        (REAL_BY_KEY[a] ? REAL.indexOf(REAL_BY_KEY[a]) : 99) - (REAL_BY_KEY[b] ? REAL.indexOf(REAL_BY_KEY[b]) : 99));
    for (const task of realKeys) {
        const r = REAL_BY_KEY[task];
        (M.real[task].default || []).forEach((src, k) => eps.real.push({
            src, task, bench: "real", ep: k + 1, name: r ? r[1] : null, real: r ? r[3] : null,
        }));
    }
    for (const t of TASKS) {
        const sets = (M.robotwin || {})[t[0]];
        if (!sets) continue;
        for (const setting of ["clean", "random"]) {
            (sets[setting] || []).forEach((src, k) => eps.robotwin.push({
                src, task: t[0], cat: t[1], bench: "robotwin", setting, ep: k + 1,
                sr: setting === "clean" ? t[8] : t[9],
            }));
        }
    }
    for (const [task, sets] of Object.entries(M.robodojo || {})) {
        (sets.default || Object.values(sets)[0] || []).forEach((src, k) =>
            eps.robodojo.push({ src, task, bench: "robodojo", ep: k + 1 }));
    }
    return eps;
}

const EPS = collectEpisodes();
// the sources each part of the page shows
const WALL = ["robotwin", "real"].filter(k => EPS[k].length);
// RoboTwin episodes on the wall per task (every clip under video/Robotwin/ is a
// successful one), alternating which setting gets two of them so clean and
// randomized come out even across the 50 tasks; the real robot's chosen four per task
const PER_TASK = 3;
const onWall = e => {
    const r = REAL_BY_KEY[e.task];
    return !r || r[5].some(p => e.src.split("/").pop().startsWith(p));
};
const SHOWN = {
    real: EPS.real.filter(onWall),
    robotwin: EPS.robotwin.filter(e => {
        const clean = Math.ceil(PER_TASK / 2) - TASKS.indexOf(TASK_BY_NAME[e.task]) % 2;
        return e.ep <= (e.setting === "clean" ? clean : PER_TASK - clean);
    }),
};
const MARQUEE = ["robotwin", "robodojo"].filter(k => EPS[k].length);
const clipCount = benches => benches.reduce((n, k) => n + EPS[k].length, 0);
const taskCount = list => new Set(list.map(e => e.task)).size;

function clipCaption(e) {
    const where = e.bench === "robotwin"
        ? `RoboTwin&nbsp;2.0, ${e.setting === "random" ? "randomized" : "clean"} scene` : BENCH[e.bench];
    const sr = e.sr != null ? ` &middot; task success ${e.sr}% (GPT-6.1-Sol, 10 episodes)`
        : e.real != null ? ` &middot; ${e.real}/10 successful on the real robot` : "";
    return `<b>${pretty(e.task)}</b> &middot; ${where} &middot; episode ${e.ep}${sr}`;
}

function tileHTML(e, i, wide) {
    return `<button type="button" class="tile${wide ? " wide" : ""}" data-i="${i}" data-task="${e.task}"
        data-src="${e.src}" style="--i:${i % 30}" aria-label="${pretty(e.task)}, episode ${e.ep}: watch large">
        <img alt="" data-src="${posterOf(e.src)}"><span class="tile-name">${pretty(e.task)}</span></button>`;
}

// one RoboTwin task, its episodes stacked: the top one plays, and its tag says
// which setting it is in
function deckHTML(deck, d) {
    const e = deck[0];
    const dots = deck.length > 1
        ? `<span class="tile-dots" aria-hidden="true">${deck.map((_, j) => `<i${j ? "" : ' class="on"'}></i>`).join("")}</span>` : "";
    return `<button type="button" class="tile deck" data-d="${d}" data-k="0" data-task="${e.task}" data-src="${e.src}"
        style="--i:${d % 30}" aria-label="${pretty(e.task)}: ${deck.length} rollouts, click for the next">
        <img alt="" data-src="${posterOf(e.src)}"><span class="tile-tag ${e.setting}">${e.setting}</span>
        <span class="tile-sr" title="Task success, GPT-6.1-Sol, 10 episodes">SR ${e.sr}%</span>
        <span class="tile-name">${pretty(e.task)}</span>${dots}</button>`;
}

// deal the next episode of a deck onto its tile
function deal(tile, deck) {
    const k = (+tile.dataset.k + 1) % deck.length, e = deck[k];
    tile.dataset.k = k;
    tile.dataset.src = e.src;
    const img = tile.querySelector("img");
    img.dataset.src = img.src = posterOf(e.src);
    const tag = tile.querySelector(".tile-tag");
    tag.className = `tile-tag ${e.setting}`;
    tag.textContent = e.setting;
    tile.querySelector(".tile-sr").textContent = `SR ${e.sr}%`;
    tile.querySelectorAll(".tile-dots i").forEach((dot, j) => dot.classList.toggle("on", j === k));
    const v = tile.querySelector("video");
    if (v) {
        v.classList.remove("is-live");
        v.addEventListener("playing", () => v.classList.add("is-live"), { once: true });
        v.src = e.src;
        const p = v.play();
        if (p) p.catch(() => {});
    }
    tile.classList.remove("dealt");
    void tile.offsetWidth;  // restart the animation
    tile.classList.add("dealt");
}

function phTileHTML(e) {
    return `<div class="tile wide is-ph" aria-hidden="true"><img alt="" data-src="${posterOf(e.src)}">
        <span class="badge-ph">PLACEHOLDER</span>
        <span class="ph-note"><span>Footage coming soon<small>blurred stand-in</small></span></span></div>`;
}

// posters load as tiles come within reach of the scroll box; clips play only
// while a tile is on screen, and are dropped once it is far away again
function watchTiles(panel, box) {
    const play = tile => {
        let v = tile.querySelector("video");
        if (!v) {
            v = document.createElement("video");
            v.muted = true;
            v.loop = true;
            v.playsInline = true;
            v.preload = "auto";
            v.src = tile.dataset.src;
            v.addEventListener("playing", () => v.classList.add("is-live"), { once: true });
            tile.insertBefore(v, tile.children[1]);
        }
        const p = v.play();
        if (p) p.catch(() => {});
    };
    const drop = tile => {
        const v = tile.querySelector("video");
        if (!v) return;
        v.pause();
        v.removeAttribute("src");
        v.load();
        v.remove();
    };
    const near = new IntersectionObserver(es => {
        for (const e of es) {
            const img = e.target.querySelector("img");
            if (e.isIntersecting) { if (!img.getAttribute("src")) img.src = img.dataset.src; } else drop(e.target);
        }
    }, { root: box, rootMargin: "500px 0px" });
    const seen = new IntersectionObserver(es => {
        for (const e of es) {
            if (e.isIntersecting) play(e.target);
            else { const v = e.target.querySelector("video"); if (v) v.pause(); }
        }
    }, { threshold: .35 });
    panel.querySelectorAll(".tile").forEach(t => {
        near.observe(t);
        if (!t.classList.contains("is-ph")) seen.observe(t);
    });
    return () => {
        near.disconnect();
        seen.disconnect();
        panel.querySelectorAll(".tile").forEach(drop);
    };
}

function buildRollouts() {
    const host = $("rollWall");
    const benches = WALL;
    if (!host || !clipCount(WALL)) return;

    host.innerHTML = `
      <div class="ro-tabs" role="tablist" aria-label="Benchmark">${benches.map(k =>
        `<button type="button" class="ro-tab" role="tab" data-k="${k}"><span class="dot"></span>${BENCH[k]}<span class="n">${SHOWN[k].length}</span></button>`).join("")}</div>
      <div class="ro-bar" id="roBar"></div>
      <p class="ro-note" id="roNote"></p>
      <div id="roPanel"></div>`;
    const bar = $("roBar"), note = $("roNote"), panel = $("roPanel");
    const state = { bench: benches[0], cat: "all", setting: "all" };
    let stop = null, shown = [], decks = [];

    function realPanel() {
        const real = SHOWN.real;
        const names = [...new Set(real.map(e => e.task))];
        // tasks with a folder but no clips yet, then unnamed slots up to REAL_TASKS
        const pending = Object.keys(M.real || {}).filter(t => !names.includes(t)).map(pretty);
        while (names.length + pending.length < REAL_TASKS) pending.push(`Task ${names.length + pending.length + 1}`);
        shown = real;
        let html = "";
        names.forEach(task => {
            const list = real.filter(e => e.task === task);
            const r = REAL_BY_KEY[task];
            html += `<div class="real-group"><div class="real-head"><h4>${pretty(task)}</h4><span>${list.length} rollouts${
                r ? ` &middot; ${r[3]}/10 successful` : ""}</span></div>
                <div class="real-grid">${list.map(e => tileHTML(e, real.indexOf(e), true)).join("")}</div></div>`;
        });
        pending.forEach((label, k) => {
            const borrowed = Array.from({ length: PH_TILES }, (_, j) => EPS.real[(k * PH_TILES + j) % EPS.real.length]);
            html += `<div class="real-group"><div class="real-head"><h4>${label}</h4><span>footage coming soon</span></div>
                <div class="real-grid">${borrowed.map(phTileHTML).join("")}</div></div>`;
        });
        const all = names.length + pending.length;
        note.innerHTML = `A dual-arm PiPER running memories rehearsed on a simulated ALOHA, rewritten by an adapter.${
            pending.length ? ` Footage for ${names.length} of ${all} tasks is in; ${pending.length === 1
            ? "the last one shows a blurred placeholder until its clips are added."
            : `the other ${pending.length} show blurred placeholders until their clips are added.`}` : ""}`;
        bar.innerHTML = `<span class="showing">${real.length} rollouts &middot; ${names.length} task${names.length > 1 ? "s" : ""}</span>`;
        return html;
    }

    function simPanel(bench) {
        let list = SHOWN[bench] || EPS[bench];
        if (bench === "robotwin") {
            const inSetting = list.filter(e => state.setting === "all" || e.setting === state.setting);
            const inCat = k => inSetting.filter(e => k === "all" || e.cat === k);
            const cats = [["all", "All"], ...CATS].map(([k, l]) => [k, l, taskCount(inCat(k))]);
            list = inCat(state.cat);
            decks = [...new Set(list.map(e => e.task))].map(t => list.filter(e => e.task === t));
            bar.innerHTML = `
                <div class="cat-chips" role="group" aria-label="Task category">${cats.map(([k, l, n]) =>
                    `<button type="button" class="cat-chip${k === state.cat ? " is-on" : ""}" data-cat="${k}">${l}<span class="n">${n}</span></button>`).join("")}</div>
                <div class="seg" role="group" aria-label="Scene">${[["all", "All scenes"], ["clean", "Clean"], ["random", "Randomized"]].map(([k, l]) =>
                    `<button type="button" class="${k === state.setting ? "is-on" : ""}" data-setting="${k}" aria-pressed="${k === state.setting}">${l}</button>`).join("")}</div>
                <span class="showing">${decks.length} tasks &middot; ${list.length} rollouts</span>`;
            note.innerHTML = `${taskCount(EPS.robotwin)} tasks, ${PER_TASK} successful episodes each: click a tile for the next.
                Randomized scenes are never rehearsed. SR: success of GPT-6.1-Sol over 10 episodes.`;
            shown = list;
            return `<div class="wall"><div class="wall-grid decks">${decks.map(deckHTML).join("")}</div></div>`;
        } else {
            bar.innerHTML = `<span class="showing">${list.length} rollouts &middot; ${taskCount(list)} tasks</span>`;
            note.innerHTML = `${taskCount(list)} RoboDojo tasks, ${Math.round(list.length / taskCount(list))} episodes each.`;
        }
        shown = list;
        return `<div class="wall"><div class="wall-grid">${list.map((e, i) => tileHTML(e, i, false)).join("")}</div></div>`;
    }

    function render() {
        if (stop) stop();
        host.querySelectorAll(".ro-tab").forEach(b => {
            const on = b.dataset.k === state.bench;
            b.classList.toggle("is-on", on);
            b.setAttribute("aria-selected", String(on));
        });
        panel.innerHTML = state.bench === "real" ? realPanel() : simPanel(state.bench);
        stop = watchTiles(panel, panel.querySelector(".wall"));
    }

    host.querySelector(".ro-tabs").addEventListener("click", e => {
        const b = e.target.closest(".ro-tab");
        if (b && b.dataset.k !== state.bench) { state.bench = b.dataset.k; state.cat = "all"; state.setting = "all"; render(); }
    });
    bar.addEventListener("click", e => {
        const c = e.target.closest("[data-cat]"), s = e.target.closest("[data-setting]");
        if (c) state.cat = c.dataset.cat;
        else if (s) state.setting = s.dataset.setting;
        else return;
        render();
    });
    panel.addEventListener("click", e => {
        const d = e.target.closest(".tile[data-d]");
        if (d) { deal(d, decks[+d.dataset.d]); return; }
        const t = e.target.closest(".tile[data-i]");
        if (!t || !window.RC) return;
        RC.lightbox.open(shown.map(ep => ({ type: "video", src: ep.src, caption: clipCaption(ep) })), +t.dataset.i);
    });

    render();
}

/* ---------------- real-robot rows ----------------
   One row per task, its name beside its three chosen clips (REAL); they load and
   play, looping, while the row is on screen, and a click opens the three large. */

function buildRealTasks() {
    const host = $("realTasks");
    if (!host || !EPS.real.length) return;
    const rows = REAL.map(r => [r, r[4].map(p => EPS.real.find(e => e.task === r[0] && e.src.split("/").pop().startsWith(p)))
        .filter(Boolean)]).filter(([, list]) => list.length);
    host.innerHTML = rows.map(([r, list], i) => `
        <div class="real-row reveal" style="--rise-delay:${i * .07}s">
            <div class="real-label"><b>${r[1]}</b><span>${r[2]}</span></div>
            <div class="real-trio">${list.map((e, k) => `<button type="button" class="real-clip" data-k="${k}"
                aria-label="${r[1]}, rollout ${k + 1} of ${list.length}: watch large"><img alt="" src="${posterOf(e.src)}"
                loading="lazy"><video muted loop playsinline preload="none"></video></button>`).join("")}</div>
        </div>`).join("");
    host.querySelectorAll(".reveal").forEach(el => riseIO.observe(el));
    host.querySelectorAll(".real-row").forEach((row, i) => {
        const list = rows[i][1];
        const videos = [...row.querySelectorAll("video")];
        videos.forEach(v => v.addEventListener("playing", () => v.classList.add("is-live"), { once: true }));
        new IntersectionObserver(es => {
            videos.forEach((v, k) => {
                if (!es[0].isIntersecting) { v.pause(); return; }
                if (!v.getAttribute("src")) v.src = list[k].src;
                const p = v.play();
                if (p) p.catch(() => {});
            });
        }, { threshold: .3 }).observe(row);
        row.querySelector(".real-trio").addEventListener("click", e => {
            const b = e.target.closest(".real-clip");
            if (b && window.RC) RC.lightbox.open(list.map(ep => ({ type: "video", src: ep.src, caption: clipCaption(ep) })), +b.dataset.k);
        });
    });
}

/* ---------------- analysis figures, one at a time ----------------
   Every figure stays in the page (so the lightbox can step through all of them);
   the tabs only choose which one shows. */

function wireFigTabs() {
    const tabs = $("figTabs");
    if (!tabs) return;
    const panes = [...document.querySelectorAll(".fig-pane")];
    const show = k => {
        tabs.querySelectorAll("button").forEach(b => {
            const on = b.dataset.k === k;
            b.classList.toggle("is-on", on);
            b.setAttribute("aria-selected", String(on));
        });
        panes.forEach(p => { p.hidden = p.dataset.k !== k; });
    };
    tabs.addEventListener("click", e => { const b = e.target.closest("button"); if (b) show(b.dataset.k); });
    show(tabs.querySelector("button").dataset.k);
}

/* ---------------- marquee under the hero ----------------
   A fixed shuffle of simulation posters (RoboTwin 2.0 and RoboDojo), in two rows
   that scroll opposite ways; each row is written twice so the loop has no seam. */

function buildMarquee() {
    const r1 = $("mq1"), r2 = $("mq2");
    if (!r1 || !clipCount(MARQUEE)) return;
    // thin the bigger sets so every source gets a fair share of the 48 posters
    const every = { real: 1, robotwin: 6, robodojo: 3 };
    const pool = MARQUEE.flatMap(k => EPS[k].filter((_, i) => i % every[k] === 0));
    let s = 7;
    const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
    for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(rnd() * (i + 1));
        [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const row = list => {
        const html = list.map(e => `<img src="${posterOf(e.src)}" alt="" decoding="async">`).join("");
        return html + html;
    };
    const half = Math.min(24, Math.floor(pool.length / 2));
    r1.innerHTML = row(pool.slice(0, half));
    r2.innerHTML = row(pool.slice(half, half * 2));
    r1.setAttribute("aria-hidden", "true");
    r2.setAttribute("aria-hidden", "true");
}

/* ---------------- BibTeX copy ---------------- */

function wireCopy() {
    const btn = $("copyBib");
    if (!btn) return;
    btn.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText($("bibText").textContent);
            btn.textContent = "Copied";
        } catch {
            btn.textContent = "Select and copy";
        }
        setTimeout(() => { btn.textContent = "Copy"; }, 1600);
    });
}

/* ---------------- boot ---------------- */

renderMath(document.body);
document.querySelectorAll(".reveal").forEach(el => riseIO.observe(el));
explainer([...document.querySelectorAll("#stageChips .chip")], $("stageDesc"), { flash: true });
buildMainTable();
buildMarquee();
buildRealTasks();
wireFigTabs();
buildRollouts();
wireCopy();
