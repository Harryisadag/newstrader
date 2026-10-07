// Portfolio tab, Trade Log tab, kill switch and live-trading controls.
document.addEventListener("alpine:init", () => {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  Alpine.data("portfolioTab", () => ({
    fmt: NT.fmt,
    data: null,
    period: "1M",
    chart: null,
    loading: false,
    busy: {},

    init() {
      const reload = () => { if (Alpine.store("nt").tab === "portfolio") this.load(); };
      window.addEventListener("nt:tab", (e) => { if (e.detail === "portfolio") { this.load(); this.loadHistory(); } });
      window.addEventListener("nt:orders_changed", reload);
      window.addEventListener("nt:account", () => { if (Alpine.store("nt").tab === "portfolio") this.load(true); });
      if (Alpine.store("nt").tab === "portfolio") { this.load(); this.loadHistory(); }
    },

    async load(quiet = false) {
      if (!quiet) this.loading = true;
      try { this.data = await NT.api.get("/portfolio"); }
      catch (e) { if (!quiet) Alpine.store("nt").error(e, "Couldn't load portfolio"); }
      finally { this.loading = false; }
    },

    get totalUnrealized() {
      if (!this.data) return 0;
      return this.data.positions.reduce((s, p) => s + (p.unrealized_pl || 0), 0);
    },

    async loadHistory() {
      try {
        const h = await NT.api.get(`/portfolio/history?period=${this.period}`);
        this.drawChart(h);
      } catch (e) { /* no keys yet - chart stays empty */ }
    },

    drawChart(h) {
      const el = this.$refs.equityChart;
      if (!el) return;
      const labels = (h.timestamp || []).map((t) => {
        const d = new Date(t);
        return this.period === "1D" ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
          : d.toLocaleDateString([], { month: "short", day: "numeric" });
      });
      const values = h.equity || [];
      const up = values.length > 1 && values[values.length - 1] >= values[0];
      const color = up ? css("--good") : css("--bad");
      if (this.chart) this.chart.destroy();
      this.chart = new Chart(el, {
        type: "line",
        data: { labels, datasets: [{ data: values, borderColor: color, backgroundColor: color + "22", fill: true, pointRadius: 0, tension: 0.25, borderWidth: 2, spanGaps: true }] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => NT.fmt.money(c.parsed.y) } } },
          scales: {
            x: { ticks: { color: css("--muted"), maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.04)" } },
            y: { ticks: { color: css("--muted"), callback: (v) => NT.fmt.money(v, 0) }, grid: { color: "rgba(255,255,255,0.06)" } },
          },
        },
      });
    },

    setPeriod(p) { this.period = p; this.loadHistory(); },

    async closePosition(p) {
      if (!confirm(`Sell ALL ${p.qty} ${p.symbol} at market now? Its stop-loss / take-profit orders will be cancelled.`)) return;
      this.busy[p.symbol] = true;
      try { await NT.api.post(`/positions/${p.symbol}/close`); Alpine.store("nt").toast("success", `Closing ${p.symbol}`); await this.load(); }
      catch (e) { Alpine.store("nt").error(e, `Couldn't close ${p.symbol}`); }
      finally { this.busy[p.symbol] = false; }
    },

    async cancelOrder(o) {
      if (!confirm(`Cancel this ${o.side} order for ${o.symbol}?`)) return;
      try { await NT.api.post(`/orders/${o.id}/cancel`); await this.load(); }
      catch (e) { Alpine.store("nt").error(e, "Couldn't cancel"); }
    },

    // Open entry orders, plus the still-working exit legs of filled bracket orders, one row each.
    get openRows() {
      const rows = [];
      const open = ["new", "accepted", "held", "pending_new", "partially_filled"];
      for (const o of (this.data?.open_orders || [])) {
        if (open.includes(o.status)) rows.push({ ...o, kind: "entry" });
        for (const l of (o.legs || [])) if (open.includes(l.status)) rows.push({ ...l, kind: "exit leg" });
      }
      return rows;
    },

    legSummary(o) {
      const legs = o.legs || [];
      return legs.map((l) => (l.order_type === "limit" ? "target " + NT.fmt.money(l.limit_price) : "stop " + NT.fmt.money(l.stop_price))).join(" · ");
    },
  }));

  Alpine.data("tradesTab", () => ({
    fmt: NT.fmt,
    trades: [],
    symbol: "",
    showLegs: true,
    loading: false,
    expanded: null,

    init() {
      window.addEventListener("nt:tab", (e) => { if (e.detail === "trades") this.load(); });
      window.addEventListener("nt:orders_changed", () => { if (Alpine.store("nt").tab === "trades") this.load(); });
      if (Alpine.store("nt").tab === "trades") this.load();
    },

    async load() {
      this.loading = true;
      try { this.trades = (await NT.api.get(`/trades?limit=1000&symbol=${encodeURIComponent(this.symbol)}`)).trades; }
      catch (e) { Alpine.store("nt").error(e, "Couldn't load trades"); }
      finally { this.loading = false; }
    },

    get visible() {
      return this.showLegs ? this.trades : this.trades.filter((t) => !["take_profit", "stop_loss"].includes(t.intent) || t.status === "filled");
    },

    get totals() {
      const filled = this.trades.filter((t) => t.realized_pl !== null && t.realized_pl !== undefined);
      const pl = filled.reduce((s, t) => s + t.realized_pl, 0);
      const wins = filled.filter((t) => t.realized_pl > 0).length;
      return { closed: filled.length, pl, winRate: filled.length ? (100 * wins) / filled.length : null };
    },

    intentLabel(i) {
      return { open_long: "Buy", open_short: "Short", close_long: "Sell", close_short: "Cover", take_profit: "Take-profit", stop_loss: "Stop-loss", external: "Other" }[i] || i;
    },
    statusClass(s) {
      if (s === "filled") return "good";
      if (["canceled", "expired", "rejected", "replaced"].includes(s)) return "";
      if (s === "rejected") return "bad";
      return "info";
    },

    async exportCsv() {
      try {
        const r = await NT.api.post("/trades/export");
        Alpine.store("nt").toast("success", `Exported ${r.rows} rows`, r.path, 10000);
        await NT.api.post("/system/open-folder", { which: "exports" }).catch(() => {});
      } catch (e) { Alpine.store("nt").error(e, "Export failed"); }
    },
    downloadUrl() { return NT.api.url("/trades/export.csv"); },
  }));

  // Header kill switch + modal
  Alpine.data("killSwitch", () => ({
    open: false,
    closePositions: false,
    busy: false,
    get engaged() { return Alpine.store("nt").status.kill_switch && Alpine.store("nt").status.kill_switch.engaged; },
    async engage() {
      this.busy = true;
      try {
        const r = await NT.api.post("/kill", { close_positions: this.closePositions });
        Alpine.store("nt").status.kill_switch = { engaged: true };
        if (r.errors && r.errors.length) Alpine.store("nt").toast("error", "Some orders may still be open", r.errors.join("\n"), 0);
        this.open = false;
      } catch (e) { Alpine.store("nt").error(e, "Kill switch failed - check Alpaca directly!"); }
      finally { this.busy = false; }
    },
    async rearm() {
      if (!confirm("Re-arm trading? NewsTrader will start placing trades again.")) return;
      try { await NT.api.post("/rearm"); Alpine.store("nt").status.kill_switch = { engaged: false }; }
      catch (e) { Alpine.store("nt").error(e); }
    },
  }));

  // Settings -> Live trading
  Alpine.data("liveTrading", () => ({
    info: { mode: "paper", has_live_keys: false, phrase: "" },
    enable: false,
    phrase: "",
    busy: false,
    async init() {
      await this.load();
      window.addEventListener("nt:mode", () => this.load());
    },
    async load() { try { this.info = await NT.api.get("/live"); } catch (e) { /* ignore */ } },
    async arm() {
      this.busy = true;
      try {
        await NT.api.post("/live/arm", { enable: this.enable, phrase: this.phrase });
        this.phrase = ""; this.enable = false;
        await this.load();
        Alpine.store("nt").toast("error", "LIVE TRADING ON", "Real money is now at risk until you switch back or restart.", 15000);
      } catch (e) { Alpine.store("nt").error(e, "Live trading stays OFF"); }
      finally { this.busy = false; }
    },
    async disarm() {
      try { await NT.api.post("/live/disarm"); await this.load(); Alpine.store("nt").toast("success", "Back to PAPER mode"); }
      catch (e) { Alpine.store("nt").error(e); }
    },
  }));
});
