// Shared helpers: header/footer, theme, formatting, data loading.
(function () {
  const THEME_KEY = "f1pace-theme";

  function applyTheme(t) {
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  try { applyTheme(localStorage.getItem(THEME_KEY)); } catch (e) {}

  function currentTheme() {
    const set = document.documentElement.getAttribute("data-theme");
    if (set) return set;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  const NAV = [
    ["index.html", "Top 100"],
    ["search.html", "Drivers"],
    ["compare.html", "Compare"],
    ["methodology.html", "Method"],
  ];

  function header(active) {
    const el = document.createElement("header");
    el.className = "site-header";
    el.innerHTML = `
      <div class="wrap">
        <a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true"></span><span>Teammate Delta</span></a>
        <nav class="nav">${NAV.map(([h, l]) => `<a href="${h}"${h === active ? ' aria-current="page"' : ""}>${l}</a>`).join("")}</nav>
        <button class="theme-toggle" type="button" aria-label="Toggle colour theme" title="Toggle theme"></button>
      </div>`;
    const btn = el.querySelector(".theme-toggle");
    const paint = () => { btn.textContent = currentTheme() === "dark" ? "☀" : "☾"; };
    btn.addEventListener("click", () => {
      const next = currentTheme() === "dark" ? "light" : "dark";
      applyTheme(next);
      try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
      paint();
    });
    paint();
    document.body.prepend(el);
  }

  function footer(meta) {
    const el = document.createElement("footer");
    el.className = "site-footer";
    el.innerHTML = `<div class="wrap">Qualifying data from <a href="https://github.com/f1db/f1db">F1DB</a> (CC BY 4.0), 1950–${meta ? meta.lastSeason : "present"}.
      ${meta ? `Model built ${meta.generated} from ${meta.comparisons.toLocaleString()} teammate qualifying comparisons.` : ""}
      Not affiliated with Formula 1.</div>`;
    document.body.append(el);
  }

  const cache = {};
  function getJSON(path) {
    if (!cache[path]) {
      cache[path] = fetch(path).then((r) => {
        if (!r.ok) throw new Error(`${path}: ${r.status}`);
        return r.json();
      });
    }
    return cache[path];
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function flag(nat) {
    if (!nat || !nat.code) return "";
    return `<img class="flag" src="https://flagcdn.com/w40/${esc(nat.code)}.png" alt="${esc(nat.name)}" title="${esc(nat.demonym || nat.name)}" loading="lazy" onerror="this.remove()">`;
  }

  const LAP = 90;
  // Signed percent; negative = faster (timing-screen convention).
  function pct(v, digits = 3) {
    if (v == null || isNaN(v)) return "—";
    const r = Math.abs(v).toFixed(digits);
    const s = +r === 0 ? "" : v < 0 ? "−" : "+";
    return `${s}${r}%`;
  }
  function secs(v, lap = LAP, digits = 3) {
    if (v == null || isNaN(v)) return "—";
    const x = (v / 100) * lap;
    const r = Math.abs(x).toFixed(digits);
    const s = +r === 0 ? "" : x < 0 ? "−" : "+";
    return `${s}${r}s`;
  }
  function cls(v) {
    return v < 0 ? "faster" : v > 0 ? "slower" : "";
  }
  function lapTime(ms) {
    if (!ms) return "—";
    const m = Math.floor(ms / 60000);
    const s = ((ms % 60000) / 1000).toFixed(3).padStart(6, "0");
    return m ? `${m}:${s}` : s;
  }
  function years(a, b) {
    if (!a) return "";
    return a === b ? String(a) : `${a}–${b}`;
  }
  function driverURL(id) {
    return `driver.html?id=${encodeURIComponent(id)}`;
  }

  window.F1 = { header, footer, getJSON, esc, flag, pct, secs, cls, lapTime, years, driverURL, LAP };
})();
