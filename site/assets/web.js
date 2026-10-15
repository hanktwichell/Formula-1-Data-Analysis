// The teammate web: graph loading, chain finding and the radial ego-network chart.
(function () {
  let graphP = null;

  function load() {
    if (!graphP) {
      graphP = F1.getJSON("data/web.json").then((w) => {
        const nodes = w.nodes.map(([id, name, peak, rank, first, last, career, main], i) =>
          ({ i, id, name, peak, rank, first, last, career, main: !!main }));
        const byId = {};
        nodes.forEach((n) => (byId[n.id] = n));
        const adj = nodes.map(() => []);
        w.edges.forEach(([a, b, n, raw, adjGap, y0, y1]) => {
          // gap stored as a minus b; keep it oriented from each side
          adj[a].push({ to: b, n, raw, adj: adjGap, y0, y1 });
          adj[b].push({ to: a, n, raw: -raw, adj: -adjGap, y0, y1 });
        });
        return { nodes, byId, adj };
      });
    }
    return graphP;
  }

  // Strongest chain between two drivers: short, and built on long pairings.
  // Edge cost 1 + 4/sqrt(races) prefers a 3-link chain of full seasons over a
  // 2-link chain through a one-off substitute.
  function chain(g, fromId, toId) {
    const s = g.byId[fromId], t = g.byId[toId];
    if (!s || !t) return null;
    const N = g.nodes.length, dist = new Float64Array(N).fill(Infinity), prev = new Int32Array(N).fill(-1), via = new Array(N);
    const done = new Uint8Array(N);
    dist[s.i] = 0;
    // binary heap
    const heap = [[0, s.i]];
    const push = (x) => { heap.push(x); let k = heap.length - 1; while (k) { const p = (k - 1) >> 1; if (heap[p][0] <= heap[k][0]) break; [heap[p], heap[k]] = [heap[k], heap[p]]; k = p; } };
    const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let k = 0; for (;;) { const l = 2 * k + 1, r = l + 1; let m = k; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === k) break; [heap[m], heap[k]] = [heap[k], heap[m]]; k = m; } } return top; };
    while (heap.length) {
      const [d, u] = pop();
      if (done[u]) continue;
      done[u] = 1;
      if (u === t.i) break;
      for (const e of g.adj[u]) {
        const nd = d + 1 + 4 / Math.sqrt(e.n);
        if (nd < dist[e.to]) { dist[e.to] = nd; prev[e.to] = u; via[e.to] = e; push([nd, e.to]); }
      }
    }
    if (!isFinite(dist[t.i])) return [];
    const path = [];
    for (let v = t.i; v !== s.i; v = prev[v]) path.push({ from: g.nodes[prev[v]], to: g.nodes[v], e: via[v] });
    return path.reverse();
  }

  function surname(n) { const p = n.split(" "); return p.length > 1 && p[p.length - 1].length <= 3 ? p.slice(-2).join(" ") : p[p.length - 1]; }

  // Radial ego network: driver in the centre, teammates on ring 1, their
  // other teammates on ring 2.
  function drawEgo(el, g, centerId) {
    const c = g.byId[centerId];
    if (!c) return false;
    const ring1 = g.adj[c.i].slice().sort((a, b) => a.y0 - b.y0 || b.n - a.n);
    if (!ring1.length) return false;
    const inR1 = new Set(ring1.map((e) => e.to));
    const ring2 = [];
    const seen = new Set([c.i, ...inR1]);
    const per = Math.max(2, Math.min(5, Math.floor(60 / ring1.length)));
    ring1.forEach((e1, k) => {
      const kids = g.adj[e1.to].filter((e) => !seen.has(e.to)).sort((a, b) => b.n - a.n).slice(0, per);
      kids.forEach((e) => { seen.add(e.to); ring2.push({ parent: k, e }); });
    });

    const W = 760, H = 760, cx = W / 2, cy = H / 2, R1 = 175, R2 = 320;
    const ang1 = ring1.map((_, k) => -Math.PI / 2 + (2 * Math.PI * k) / ring1.length);
    const pos = {};
    pos[c.i] = [cx, cy];
    ring1.forEach((e, k) => (pos[e.to] = [cx + R1 * Math.cos(ang1[k]), cy + R1 * Math.sin(ang1[k])]));
    // spread ring-2 nodes evenly around the circle, in parent order, so they sit near their parent
    ring2.forEach((r, k) => {
      const a = -Math.PI / 2 + (2 * Math.PI * (k + 0.5)) / Math.max(ring2.length, 1) - Math.PI / Math.max(ring1.length, 1);
      r.a = a;
      pos[r.e.to] = [cx + R2 * Math.cos(a), cy + R2 * Math.sin(a)];
    });

    const rad = (n) => 4 + Math.min(10, Math.sqrt(n) * 1.3);
    const color = (n) => {
      if (n.peak == null) return "var(--faint)";
      return n.peak < 0 ? "var(--faster)" : "var(--slower)";
    };
    const op = (n) => Math.min(1, 0.35 + Math.abs(n.peak) * 1.6);
    let edges = "", nodes = "", labels = "";
    // ring-2 edges first (behind)
    ring2.forEach((r) => {
      // link to every ring-1 teammate this driver shared a car with
      g.adj[r.e.to].forEach((e) => {
        if (inR1.has(e.to)) {
          const [x1, y1] = pos[e.to], [x2, y2] = pos[r.e.to];
          edges += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--bar-soft)" stroke-width="${0.6 + Math.min(2.4, e.n / 20)}" />`;
        }
      });
    });
    // ring-1 cross links (teammates who were also teammates of each other)
    ring1.forEach((e1) => g.adj[e1.to].forEach((e) => {
      if (inR1.has(e.to) && e.to > e1.to) {
        const [x1, y1] = pos[e1.to], [x2, y2] = pos[e.to];
        edges += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--bar-soft)" stroke-width="${0.6 + Math.min(2.4, e.n / 20)}" />`;
      }
    }));
    ring1.forEach((e) => {
      const [x, y] = pos[e.to];
      const col = e.adj < 0 ? "var(--faster)" : e.adj > 0 ? "var(--slower)" : "var(--faint)";
      edges += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="${col}" stroke-opacity=".75" stroke-width="${1 + Math.min(6, e.n / 12)}" stroke-linecap="round"><title>${F1.esc(c.name)} vs ${F1.esc(g.nodes[e.to].name)}: ${F1.pct(e.raw)} median over ${e.n} races</title></line>`;
    });
    const node = (n, x, y, r, big) => `<a href="${F1.driverURL(n.id)}"><circle cx="${x}" cy="${y}" r="${r}" fill="${color(n)}" fill-opacity="${op(n)}" stroke="var(--surface)" stroke-width="1.5"><title>${F1.esc(n.name)} (${n.first}–${n.last}) · peak ${F1.pct(n.peak)}${n.rank ? ` · #${n.rank}` : ""}</title></circle></a>`;
    ring2.forEach((r) => {
      const n = g.nodes[r.e.to], [x, y] = pos[n.i];
      nodes += node(n, x, y, 3.5 + Math.min(6, Math.sqrt(g.adj[n.i].reduce((s, e) => s + e.n, 0)) * 0.5));
      const out = r.a, anchor = Math.cos(out) > 0.2 ? "start" : Math.cos(out) < -0.2 ? "end" : "middle";
      const lx = x + Math.cos(out) * 12, ly = y + Math.sin(out) * 12 + 4;
      labels += `<text x="${lx}" y="${ly}" text-anchor="${anchor}" class="w2">${F1.esc(surname(n.name))}</text>`;
    });
    ring1.forEach((e, k) => {
      const n = g.nodes[e.to], [x, y] = pos[n.i];
      nodes += node(n, x, y, rad(e.n) + 2);
      const a = ang1[k];
      const anchor = Math.cos(a) > 0.3 ? "start" : Math.cos(a) < -0.3 ? "end" : "middle";
      const off = rad(e.n) + 8;
      const lx = x + Math.cos(a) * off, ly = y + Math.sin(a) * off + (Math.sin(a) > 0.3 ? 12 : Math.sin(a) < -0.3 ? -4 : 4);
      labels += `<text x="${lx}" y="${ly}" text-anchor="${anchor}" class="w1">${F1.esc(n.name)}</text>`;
    });
    nodes += `<circle cx="${cx}" cy="${cy}" r="20" fill="var(--text)"/>`;
    labels += `<text x="${cx}" y="${cy + 40}" text-anchor="middle" class="w0">${F1.esc(c.name)}</text>`;

    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Teammate web around ${F1.esc(c.name)}">
      <style>.w0{font:700 17px var(--sans);fill:var(--text)}.w1{font:600 14px var(--sans);fill:var(--text)}.w2{font:400 12px var(--sans);fill:var(--muted)}</style>
      <circle cx="${cx}" cy="${cy}" r="${R1}" fill="none" stroke="var(--border)" stroke-dasharray="2 4"/>
      <circle cx="${cx}" cy="${cy}" r="${R2}" fill="none" stroke="var(--border)" stroke-dasharray="2 4"/>
      ${edges}${nodes}${labels}</svg>`;
    return { ring1: ring1.length, ring2: ring2.length };
  }

  window.F1Web = { load, chain, drawEgo };
})();
