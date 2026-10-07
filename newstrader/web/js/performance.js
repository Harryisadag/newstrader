// Performance tab (is the AI right?) and Backtest tab.
document.addEventListener("alpine:init", () => {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  Alpine.data("performanceTab", () => ({
    fmt: NT.fmt,
    horizon: "1h",
    days: 30,
    data: null,
    chart: null,

    init() {
      window.addEventListener("nt:tab", (e) => { if (e.detail === "performance") this.load(); });
      window.addEventListener("nt:performance_updated", () => { if (Alpine.store("nt").tab === "performance") this.load(); });
      if (Alpine.store("nt").tab === "performance") this.load();
    },

    async load() {
      try {
        this.data = await NT.api.get(`/performance?horizon=${this.horizon}&days=${this.days}`);
        this.$nextTick(() => this.draw());
      } catch (e) { Alpine.store("nt").error(e, "Couldn't load performance"); }
    },

    draw() {
      const el = this.$refs.bucketChart;
      if (!el || !this.data) return;
      const b = this.data.by_confidence;
      if (this.chart) this.chart.destroy();
      this.chart = new Chart(el, {
        type: "bar",
        data: {
          labels: b.map((x) => x.key),
          datasets: [{
            label: "Win rate %",
            data: b.map((x) => x.win_rate),
            backgroundColor: b.map((x) => (x.win_rate === null ? css("--panel-3") : x.win_rate >= 55 ? css("--good") : x.win_rate >= 45 ? css("--warn") : css("--bad"))),
            borderRadius: 6,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: (c) => { const x = b[c.dataIndex]; return x.count ? `${x.win_rate}% right (${x.wins}/${x.count}), avg move ${x.avg_return}%` : "no data yet"; } } },
          },
          scales: {
            y: { min: 0, max: 100, ticks: { color: css("--muted"), callback: (v) => v + "%" }, grid: { color: "rgba(255,255,255,0.06)" } },
            x: { ticks: { color: css("--muted") }, grid: { display: false }, title: { display: true, text: "AI confidence", color: css("--muted") } },
          },
        },
      });
    },

    retClass(v) { return v === null || v === undefined ? "" : v > 0 ? "pos" : v < 0 ? "neg" : ""; },
    ret(v) { return v === null || v === undefined ? "…" : NT.fmt.pct(v); },
    winText(g) { return g.count ? `${g.win_rate}%` : "—"; },
  }));

  Alpine.data("backtestTab", () => ({
    fmt: NT.fmt,
    form: { start: "", end: "", symbols: "", max_articles: 50, budget_usd: 3, hold: "eod", model: "" },
    estimate: null,
    runs: [],
    warning: "",
    selected: null,
    detail: null,
    chart: null,
    busy: false,

    init() {
      const d = new Date();
      const iso = (x) => x.toISOString().slice(0, 10);
      const end = new Date(d.getTime() - 86400000);
      const start = new Date(end.getTime() - 6 * 86400000);
      this.form.start = iso(start); this.form.end = iso(end);
      window.addEventListener("nt:tab", (e) => { if (e.detail === "backtest") this.loadRuns(); });
      window.addEventListener("nt:backtest", (e) => {
        const r = this.runs.find((x) => x.id === e.detail.id);
        if (r) Object.assign(r, { progress: e.detail.progress, message: e.detail.message, status: e.detail.status || r.status });
        if (e.detail.status && e.detail.status !== "running") { this.loadRuns(); if (this.selected === e.detail.id) this.open(e.detail.id); }
      });
      if (Alpine.store("nt").tab === "backtest") this.loadRuns();
    },

    async loadRuns() {
      try { const r = await NT.api.get("/backtest/runs"); this.runs = r.runs; this.warning = r.warning; }
      catch (e) { /* ignore */ }
    },

    async getEstimate() {
      try { this.estimate = await NT.api.post("/backtest/estimate", this.form); }
      catch (e) { Alpine.store("nt").error(e, "Check the backtest settings"); }
    },

    async run() {
      if (!this.estimate) await this.getEstimate();
      if (!this.estimate) return;
      const msg = `This will analyse up to ${this.estimate.articles} historical stories with ${this.estimate.model}.\n\n` +
        `Estimated Claude cost: up to about $${this.estimate.max_cost} (hard stop at your $${this.form.budget_usd} budget).\n\nRun it?`;
      if (!confirm(msg)) return;
      this.busy = true;
      try {
        const r = await NT.api.post("/backtest/run", { ...this.form, confirm: true });
        await this.loadRuns();
        this.open(r.id);
      } catch (e) { Alpine.store("nt").error(e, "Backtest didn't start"); }
      finally { this.busy = false; }
    },

    async cancel(run) { try { await NT.api.post(`/backtest/runs/${run.id}/cancel`); } catch (e) { Alpine.store("nt").error(e); } },
    async remove(run) {
      if (!confirm("Delete this backtest?")) return;
      try { await NT.api.del(`/backtest/runs/${run.id}`); if (this.selected === run.id) { this.selected = null; this.detail = null; } await this.loadRuns(); }
      catch (e) { Alpine.store("nt").error(e); }
    },

    async open(id) {
      this.selected = id;
      try { this.detail = await NT.api.get(`/backtest/runs/${id}`); this.$nextTick(() => this.drawCurve()); }
      catch (e) { Alpine.store("nt").error(e); }
    },

    drawCurve() {
      const el = this.$refs.btChart;
      const s = this.detail && this.detail.run.summary;
      if (!el || !s) return;
      const pts = s.equity_curve || [];
      if (this.chart) this.chart.destroy();
      const color = (pts.length && pts[pts.length - 1].pnl < 0) ? css("--bad") : css("--good");
      this.chart = new Chart(el, {
        type: "line",
        data: { labels: pts.map((p) => NT.fmt.dateTime(p.t)), datasets: [{ data: pts.map((p) => p.pnl), borderColor: color, backgroundColor: color + "22", fill: true, pointRadius: 2, tension: 0.2 }] },
        options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } },
          scales: { x: { ticks: { color: css("--muted"), maxTicksLimit: 8 }, grid: { display: false } },
            y: { ticks: { color: css("--muted"), callback: (v) => NT.fmt.money(v, 0) }, grid: { color: "rgba(255,255,255,0.06)" } } } },
      });
    },

    statusClass(s) { return { done: "good", running: "info", failed: "bad", cancelled: "", stopped: "" }[s] || ""; },
  }));
});
