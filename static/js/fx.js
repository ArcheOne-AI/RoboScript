/* RoboScript project page: the effects layer. Theme and motion switches, the
   reading-progress line and section highlight in the nav, back-to-top, the
   pointer spotlight on glass cards, count-up numbers, the hero's orbiting dots,
   and the lightbox that figures and rollout clips open in. main.js reaches the
   shared pieces through window.RC. */

(function () {
    const $ = id => document.getElementById(id);
    const root = document.documentElement;
    const calm = () => root.dataset.motion === "off";
    const store = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } };

    /* ---------------- theme and motion ---------------- */

    $("themeBtn").addEventListener("click", () => {
        const next = root.dataset.theme === "dark" ? "light" : "dark";
        root.dataset.theme = next;
        root.dataset.bsTheme = next;
        document.querySelector('meta[name="theme-color"]').content = next === "dark" ? "#070B14" : "#FFFFFF";
        store("rs-theme", next);
        dispatchEvent(new Event("rc-theme"));
    });
    $("motionBtn").addEventListener("click", () => {
        root.dataset.motion = calm() ? "on" : "off";
        store("rs-motion", root.dataset.motion);
        dispatchEvent(new Event("rc-motion"));
    });
    const syncMotionLabel = () => $("motionBtn").setAttribute("aria-pressed", String(!calm()));
    syncMotionLabel();
    addEventListener("rc-motion", syncMotionLabel);

    /* ---------------- progress line, back to top ---------------- */

    const bar = $("progress"), up = $("toTop");
    let ticking = false;
    function onScroll() {
        const max = document.documentElement.scrollHeight - innerHeight;
        bar.style.setProperty("--p", max > 0 ? (scrollY / max).toFixed(4) : 0);
        up.classList.toggle("is-on", scrollY > innerHeight * .9);
        ticking = false;
    }
    addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
    onScroll();
    up.addEventListener("click", () => scrollTo({ top: 0, behavior: calm() ? "auto" : "smooth" }));

    /* ---------------- section highlight in the nav ---------------- */

    const links = [...document.querySelectorAll(".nav-links a")];
    const spy = new IntersectionObserver(es => {
        for (const e of es) {
            if (!e.isIntersecting) continue;
            links.forEach(a => a.classList.toggle("is-on", a.getAttribute("href") === "#" + e.target.id));
        }
    }, { rootMargin: "-40% 0px -55% 0px" });
    links.forEach(a => { const s = document.querySelector(a.getAttribute("href")); if (s) spy.observe(s); });
    spy.observe($("top"));  // back in the hero, no link is highlighted

    /* ---------------- section menu on phones, where the links are hidden ---------------- */

    const nav = $("topnav"), menu = $("menuBtn");
    const setMenu = open => { nav.classList.toggle("is-open", open); menu.setAttribute("aria-expanded", String(open)); };
    menu.addEventListener("click", () => setMenu(!nav.classList.contains("is-open")));
    links.forEach(a => a.addEventListener("click", () => setMenu(false)));
    document.addEventListener("click", e => { if (!nav.contains(e.target)) setMenu(false); });
    addEventListener("keydown", e => { if (e.key === "Escape") setMenu(false); });

    /* ---------------- spotlight on glass cards ---------------- */

    document.addEventListener("pointermove", e => {
        const g = e.target.closest && e.target.closest(".glass");
        if (!g) return;
        const r = g.getBoundingClientRect();
        g.style.setProperty("--mx", `${e.clientX - r.left}px`);
        g.style.setProperty("--my", `${e.clientY - r.top}px`);
    }, { passive: true });

    /* ---------------- count-up numbers ----------------
       data-count is the target, data-dec its decimals, data-suffix any unit. */

    function countUp(el) {
        const to = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0), suf = el.dataset.suffix || "";
        const show = v => { el.textContent = v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suf; };
        if (isNaN(to)) return;
        if (calm()) { show(to); return; }
        const t0 = performance.now(), dur = 1400;
        const step = now => {
            const k = Math.min(1, (now - t0) / dur);
            show(to * (1 - Math.pow(1 - k, 3)));
            if (k < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    }
    const countIO = new IntersectionObserver((es, obs) => {
        for (const e of es) {
            if (!e.isIntersecting) continue;
            obs.unobserve(e.target);
            countUp(e.target);
        }
    }, { threshold: .6 });
    const watchCounts = scope => scope.querySelectorAll("[data-count]").forEach(el => countIO.observe(el));
    watchCounts(document);

    /* ---------------- hero: dots orbiting a few centres ----------------
       Each dot circles one of five centres on its own tilted ellipse, and nearby
       dots are joined by faint lines. It pauses off screen, and with motion off it
       draws one still frame. */

    (function heroDots() {
        const c = $("heroCanvas");
        if (!c || !c.getContext) return;
        const ctx = c.getContext("2d");
        let w = 0, h = 0, dots = [], pal = [], raf = 0, last = 0, inView = true;
        let px = 0, py = 0;  // pointer offset for a little parallax

        const rgb = hex => {
            const m = hex.trim().replace("#", "");
            return [0, 2, 4].map(i => parseInt(m.slice(i, i + 2), 16)).join(",");
        };
        function palette() {
            const s = getComputedStyle(root);
            pal = ["--accent", "--blue", "--teal"].map(v => rgb(s.getPropertyValue(v)));
        }
        const rand = (a, b) => a + Math.random() * (b - a);
        function seed() {
            const n = Math.round(Math.min(110, Math.max(36, w * h / 14000)));
            const centres = Array.from({ length: 5 }, (_, i) => ({
                x: w * (.1 + .2 * i) + rand(-30, 30),
                y: h * rand(.25, .7),
            }));
            dots = Array.from({ length: n }, (_, i) => {
                const cc = centres[i % centres.length];
                return {
                    cx: cc.x, cy: cc.y,
                    rx: rand(40, Math.min(300, w * .22)), ry: rand(18, Math.min(150, h * .18)),
                    rot: rand(0, Math.PI), a: rand(0, Math.PI * 2),
                    sp: rand(.05, .2) * (Math.random() < .5 ? -1 : 1),
                    r: rand(1, 2.6), col: i % 3, x: 0, y: 0,
                };
            });
        }
        function resize() {
            const dpr = Math.min(devicePixelRatio || 1, 2);
            w = c.clientWidth;
            h = c.clientHeight;
            c.width = Math.round(w * dpr);
            c.height = Math.round(h * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            seed();
        }
        function draw(dt) {
            ctx.clearRect(0, 0, w, h);
            for (const d of dots) {
                d.a += d.sp * dt;
                const ca = Math.cos(d.a), sa = Math.sin(d.a), cr = Math.cos(d.rot), sr = Math.sin(d.rot);
                d.x = d.cx + d.rx * ca * cr - d.ry * sa * sr + px;
                d.y = d.cy + d.rx * ca * sr + d.ry * sa * cr + py;
            }
            const reach = 120;
            ctx.lineWidth = 1;
            for (let i = 0; i < dots.length; i++) {
                const a = dots[i];
                for (let j = i + 1; j < dots.length; j++) {
                    const b = dots[j], dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy;
                    if (d2 > reach * reach) continue;
                    ctx.strokeStyle = `rgba(${pal[a.col]},${(.22 * (1 - Math.sqrt(d2) / reach)).toFixed(3)})`;
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.stroke();
                }
            }
            for (const d of dots) {
                ctx.fillStyle = `rgba(${pal[d.col]},.16)`;
                ctx.beginPath();
                ctx.arc(d.x, d.y, d.r * 3.2, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = `rgba(${pal[d.col]},.9)`;
                ctx.beginPath();
                ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        function loop(now) {
            const dt = Math.min(.05, (now - (last || now)) / 1000);
            last = now;
            draw(dt);
            raf = requestAnimationFrame(loop);
        }
        function sync() {
            cancelAnimationFrame(raf);
            raf = 0;
            last = 0;
            if (calm() || !inView || document.hidden) { draw(0); return; }
            raf = requestAnimationFrame(loop);
        }
        palette();
        resize();
        sync();
        new ResizeObserver(() => { resize(); if (!raf) draw(0); }).observe(c);
        new IntersectionObserver(es => { inView = es[0].isIntersecting; sync(); }).observe(c);
        document.addEventListener("visibilitychange", sync);
        addEventListener("rc-motion", sync);
        addEventListener("rc-theme", () => { palette(); if (!raf) draw(0); });
        c.parentElement.addEventListener("pointermove", e => {
            const r = c.getBoundingClientRect();
            px = (e.clientX - r.left - w / 2) * -.03;
            py = (e.clientY - r.top - h / 2) * -.03;
        }, { passive: true });
    })();

    /* ---------------- lightbox ----------------
       open(items, index): items are {type: "img" | "video", src, caption}. The
       arrows (and the arrow keys) step through the list; Esc or a click on the
       backdrop closes it. */

    const lightbox = (function () {
        const dlg = $("lightbox"), stage = $("lbStage"), cap = $("lbCap"), prev = $("lbPrev"), next = $("lbNext");
        let items = [], at = 0;
        function show() {
            const it = items[at];
            stage.classList.toggle("is-fig", it.type === "img");
            stage.innerHTML = it.type === "img"
                ? `<img src="${it.src}" alt="">`
                : `<video src="${it.src}" controls autoplay muted loop playsinline></video>`;
            cap.innerHTML = it.caption || "";
            prev.hidden = next.hidden = items.length < 2;
        }
        const step = k => { at = (at + k + items.length) % items.length; show(); };
        prev.addEventListener("click", () => step(-1));
        next.addEventListener("click", () => step(1));
        $("lbClose").addEventListener("click", () => dlg.close());
        dlg.addEventListener("close", () => { stage.innerHTML = ""; });
        dlg.addEventListener("click", e => { if (e.target === dlg) dlg.close(); });
        dlg.addEventListener("keydown", e => {
            if (e.key === "ArrowLeft") step(-1);
            if (e.key === "ArrowRight") step(1);
        });
        return {
            open(list, index) {
                items = list;
                at = index || 0;
                show();
                if (!dlg.open) dlg.showModal();
            },
        };
    })();

    // every figure opens large; the arrows step through all of them
    const figs = [...document.querySelectorAll("[data-zoom]")];
    const figItems = figs.map(f => {
        const img = f.querySelector("img");
        const after = f.nextElementSibling;
        const caption = after && after.classList.contains("fig-caption") ? after.innerHTML : img.alt;
        return { type: "img", src: img.currentSrc || img.src, caption };
    });
    figs.forEach((f, i) => {
        f.setAttribute("role", "button");
        f.tabIndex = 0;
        const open = () => lightbox.open(figItems, i);
        f.addEventListener("click", open);
        f.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    });

    window.RC = { calm, lightbox, countUp, watchCounts };
})();
