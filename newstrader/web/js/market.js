// Market tab: the whole market (S&P 500, Nasdaq... via their ETFs), world markets, sudden spikes, today's top
// movers and the stocks being watched. Live updates: "market_event" (one spike or move) and "market" (after each check).
// Clicking a stock opens its chart card (/api/chart/{symbol}): price with VWAP and the nearest support / resistance,
// indicator labels, the patterns showing now with plain explanations, and the one-line summary.
document.addEventListener("alpine:init", () => {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const FRAMES = { "1m": "1-minute chart", "5m": "5-minute chart", "1d": "daily chart" };
  const TRENDS = { up: "Uptrend", down: "Downtrend", sideways: "Sideways" };
  const PATTERN_NAMES = {
    rsi_overbought: "RSI overbought", rsi_oversold: "RSI oversold", rsi_divergence: "RSI divergence",
    macd_cross: "MACD cross", macd_zero_cross: "MACD crossed zero", vwap_reclaim: "Back above VWAP",
    vwap_lost: "Fell below VWAP", vwap_stretch: "Far from VWAP", opening_range_breakout: "Opening range breakout",
    opening_range_breakdown: "Opening range breakdown", new_high_of_day: "New high of the day",
    new_low_of_day: "New low of the day", ma_trend: "Moving-average trend", bollinger_breakout: "Bollinger squeeze",
  };
  let priceChart = null;  // the Chart.js instance (kept outside Alpine's reactive data)

  const KINDS = {
    spike_up: { label: "Jump", cls: "good" },
    spike_down: { label: "Drop", cls: "bad" },
    volume_surge: { label: "Busy trading", cls: "warn" },
  };
  const REASONS = {
    position: "You hold it", order: "Open order", watchlist: "Your watchlist", signal: "Recent signal",
    mover: "Top mover", active: "Most traded", index: "Market gauge", world: "World market",
  };
  const MAX_EVENTS = 300;

  Alpine.data("marketTab", () => ({
    fmt: NT.fmt,
    data: null,
    moversView: "gainers",
    loading: false,
    scanning: false,
    chartSym: "",
    chart: null,
    chartLoading: false,
    chartError: "",

    init() {
      window.addEventListener("nt:tab", (e) => { if (e.detail === "market") this.load(); });
      window.addEventListener("nt:market_event", (e) => this.addEvent(e.detail));
      window.addEventListener("nt:market", (e) => this.onSummary(e.detail));
      window.addEventListener("nt:open-chart", (e) => this.openChart(e.detail));
      if (Alpine.store("nt").tab === "market") this.load();
    },

    get visible() { return Alpine.store("nt").tab === "market"; },

    async load(quiet = false) {
      if (!quiet) this.loading = true;
      try { this.data = await NT.api.get("/market"); }
      catch (e) { if (!quiet) Alpine.store("nt").error(e, "Couldn't load the market"); }
      finally { this.loading = false; }
    },

    addEvent(ev) {
      if (!this.data || !ev || this.data.events.some((x) => x.id === ev.id)) return;
      this.data.events.unshift(ev);
      if (this.data.events.length > MAX_EVENTS) this.data.events.length = MAX_EVENTS;
    },

    onSummary(sum) {
      if (!this.data || !sum) return;
      const { phase, last_scan, last_refresh, indices, world, counts } = sum;
      Object.assign(this.data, { phase, last_scan, last_refresh, indices, world, counts });
      if (this.visible) this.load(true);  // movers and the watch list too
      if (this.visible && this.chartSym && !this.chartLoading) this.openChart(this.chartSym, true);  // a fresh minute
    },

    async scanNow() {
      this.scanning = true;
      try { await NT.api.post("/market/scan"); await this.load(true); }
      catch (e) { Alpine.store("nt").error(e, "Couldn't check the market"); }
      finally { this.scanning = false; }
    },

    // ---- chart card ----
    async openChart(sym, refresh = false) {
      sym = String(sym || "").trim().toUpperCase();
      if (!sym) return;
      if (sym !== this.chartSym) { this.chart = null; this.destroyChart(); }
      this.chartSym = sym;
      this.chartError = "";
      if (!refresh) this.$nextTick(() => this.$refs.chartCard && this.$refs.chartCard.scrollIntoView({ behavior: "smooth", block: "start" }));
      this.chartLoading = true;
      try {
        const data = await NT.api.get(`/chart/${encodeURIComponent(sym)}`);
        if (data.symbol !== this.chartSym) return;  // another stock was clicked meanwhile
        this.chart = data;
        this.$nextTick(() => this.drawChart());
      } catch (e) {
        if (sym === this.chartSym) { this.chartError = e.message; if (!refresh) this.chart = null; }
      } finally { this.chartLoading = false; }
    },
    closeChart() { this.chartSym = ""; this.chart = null; this.chartError = ""; this.destroyChart(); },
    destroyChart() { if (priceChart) { priceChart.destroy(); priceChart = null; } },

    drawChart() {
      const el = this.$refs.priceChart;
      const c = this.chart && Alpine.raw(this.chart);  // plain arrays for Chart.js (it watches the arrays it is given)
      if (!el || !c || !c.series.t.length) { this.destroyChart(); return; }
      const z = this.fmt.zone();
      const labels = c.series.t.map((t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", timeZone: z.timeZone }));
      const flat = (v) => labels.map(() => v);
      const lv = this.nearest;
      const datasets = [
        { label: "Price", data: c.series.close.slice(), borderColor: css("--accent"), borderWidth: 2, pointRadius: 0, tension: 0.1, spanGaps: true },
        { label: "VWAP (average price paid today)", data: c.series.vwap.slice(), borderColor: css("--warn"), borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0, spanGaps: true },
      ];
      if (lv.support) datasets.push({ label: "Support " + this.fmt.money(lv.support.price), data: flat(lv.support.price), borderColor: css("--good"), borderWidth: 1, borderDash: [2, 3], pointRadius: 0 });
      if (lv.resistance) datasets.push({ label: "Resistance " + this.fmt.money(lv.resistance.price), data: flat(lv.resistance.price), borderColor: css("--bad"), borderWidth: 1, borderDash: [2, 3], pointRadius: 0 });
      this.destroyChart();
      priceChart = new Chart(el, {
        type: "line",
        data: { labels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: { labels: { color: css("--muted"), boxWidth: 14 } },
            tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label.split(" (")[0]}: ${NT.fmt.money(ctx.parsed.y)}` } },
          },
          scales: {
            x: { ticks: { color: css("--muted"), maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.04)" } },
            y: { ticks: { color: css("--muted"), callback: (v) => NT.fmt.money(v) }, grid: { color: "rgba(255,255,255,0.06)" } },
          },
        },
      });
    },

    get chartWhen() {
      const r = this.chart && this.chart.reading;
      return r && r.bar_at ? "Last bar " + this.fmt.time(r.bar_at) : "";
    },
    get chartLevels() { return (this.chart && this.chart.reading.levels) || []; },
    // the closest support below the price and the closest resistance above it
    get nearest() {
      const lv = this.chartLevels;
      const sup = lv.filter((l) => l.kind === "support").sort((a, b) => b.price - a.price)[0] || null;
      const res = lv.filter((l) => l.kind === "resistance").sort((a, b) => a.price - b.price)[0] || null;
      return { support: sup, resistance: res };
    },
    get chartPatterns() { return (this.chart && this.chart.reading.patterns) || []; },
    get lean() {
      const sc = this.chart ? this.chart.reading.score : 0;
      const tip = `Chart score ${sc} (from -1, very bearish, to +1, very bullish): the trends, the side of VWAP and the patterns, added up.`;
      if (sc >= 0.3) return { label: "Leans bullish", cls: "good", tip };
      if (sc <= -0.3) return { label: "Leans bearish", cls: "bad", tip };
      return { label: "Mixed", cls: "", tip };
    },
    get chartChips() {
      const r = this.chart && this.chart.reading;
      if (!r) return [];
      const f = this.fmt;
      const out = [];
      if (r.rsi !== null) out.push({ label: `RSI ${Math.round(r.rsi)}`, cls: r.rsi > 80 || r.rsi < 20 ? "warn" : "",
        tip: "RSI (5-minute chart, 0-100) shows how fast the price has been moving. Over 70 = it ran up fast, under 30 = it fell fast. Over 80 (or under 20) counts as stretched: the move may already be over." });
      if (r.macd_hist !== null) out.push({ label: r.macd_hist > 0 ? "MACD: momentum up" : "MACD: momentum down", cls: r.macd_hist > 0 ? "good" : "bad",
        tip: "MACD compares a fast and a slow average of the price (5-minute chart). Above zero = the price is speeding up; below zero = slowing down or falling." });
      if (r.vwap_pct !== null) out.push({ label: `${f.pct(r.vwap_pct)} vs VWAP`, cls: (r.vwap_atr !== null && Math.abs(r.vwap_atr) >= 2) ? "warn" : (r.vwap_pct > 0 ? "good" : "bad"),
        tip: "VWAP is the average price everyone paid today, weighted by how much they bought. Above it = buyers are in control today. Far above it (2 normal days' moves or more) = stretched." });
      if (r.rel_volume !== null) out.push({ label: `Volume ${f.num(r.rel_volume, 1)}x normal`, cls: r.rel_volume >= 1.5 ? "info" : "",
        tip: `How busy trading is compared with normal (${r.rel_volume_basis}). 1.5x or more is heavy volume - moves on heavy volume are more believable.` });
      if (r.bar_rel_volume !== null && r.bar_rel_volume >= 1.5) out.push({ label: `Last 5 min: ${f.num(r.bar_rel_volume, 1)}x volume`, cls: "info",
        tip: "The latest 5-minute bar traded this many times its normal volume - something is happening right now." });
      if (TRENDS[r.trend]) out.push({ label: TRENDS[r.trend], cls: r.trend === "up" ? "good" : r.trend === "down" ? "bad" : "",
        tip: "The trend on the 5-minute chart (the price against its 20- and 50-bar averages)." });
      if (TRENDS[r.trend_daily]) out.push({ label: "Daily: " + TRENDS[r.trend_daily].toLowerCase(), cls: r.trend_daily === "up" ? "good" : r.trend_daily === "down" ? "bad" : "",
        tip: "The bigger trend on the daily chart (the last few months)." });
      if (r.atr_pct !== null) out.push({ label: `Normal day's move ${f.num(r.atr_pct, 1)}%`, cls: "",
        tip: "ATR: how much this stock usually moves in a day. A 2% move is big for a stock that normally moves 1%, and nothing special for one that moves 5%." });
      return out;
    },
    get signalNote() {
      const mode = this.chart && this.chart.signals;
      if (mode === "review") return "New chart signals go to manual review (Settings -> Charts). They are never traded by themselves.";
      if (mode === "off") return "Chart signals are off (Settings -> Charts) - this is only what the chart would say.";
      return "Watch only: recorded and scored in Performance, never traded or alerted (Settings -> Charts).";
    },
    patternName(p) {
      return PATTERN_NAMES[p.name] || (p.name.charAt(0).toUpperCase() + p.name.slice(1).replace(/_/g, " "));
    },
    frameName(tf) { return FRAMES[tf] || tf; },
    dirClass(d) { return d === "bullish" ? "good" : d === "bearish" ? "bad" : ""; },

    // ---- header row ----
    get clock() { return (this.data && this.data.market) || null; },
    get isOpen() { return !!(this.clock && this.clock.is_open); },
    get marketLabel() {
      if (!this.clock) return "—";
      if (this.isOpen) return "Open";
      return this.data.phase === "extended" ? "Extended hours" : "Closed";
    },
    get marketSub() {
      const c = this.clock;
      if (!c) return "";
      return c.is_open ? "Closes " + this.fmt.dateTime(c.next_close) : "Opens " + this.fmt.dateTime(c.next_open);
    },
    get lastCheck() {
      if (!this.data) return "";
      if (this.data.last_scan) return "Last check " + this.fmt.time(this.data.last_scan);
      if (this.data.last_refresh) return "Prices from " + this.fmt.time(this.data.last_refresh);
      return this.data.running ? "Not checked yet" : "";
    },
    get indices() { return (this.data && this.data.indices) || []; },
    get world() { return (this.data && this.data.world) || []; },
    tileName(t) { return t.name || t.symbol; },

    // ---- spikes & alerts ----
    get events() { return (this.data && this.data.events) || []; },
    kind(ev) {
      if (ev.kind === "market_move") return ev.scope === "world" ? { label: "World", cls: "info" } : { label: "Market", cls: "info" };
      return KINDS[ev.kind] || { label: ev.kind, cls: "" };
    },
    title(ev) { return (ev.detail && ev.detail.title) || `${ev.symbol} ${ev.kind}`; },
    moveText(ev) {
      if (ev.change_pct === null || ev.change_pct === undefined) return "";
      const p = this.fmt.pct(ev.change_pct, 1);
      return ev.window_min ? `${p} in ${ev.window_min} min` : `${p} today`;
    },

    // ---- top movers ----
    get moversRows() {
      if (!this.data) return [];
      if (this.moversView === "active") return this.data.most_actives.items || [];
      return this.data.movers[this.moversView] || [];
    },
    get moversEmpty() {
      if (this.data && this.data.movers.error) return this.data.movers.error;
      return this.data && this.data.running ? "No movers yet." : "Shown once the market monitor is running.";
    },

    // ---- watching ----
    get watching() { return (this.data && this.data.watching) || []; },
    get windowMin() { return (this.data && this.data.thresholds && this.data.thresholds.spike_window_minutes) || 5; },
    reasonText(r) { return REASONS[r] || r; },
    get watchingNote() {
      if (!this.data) return "";
      const c = this.data.counts || {};
      let s = `${c.watching || 0} stocks, checked for sudden spikes`;
      if (c.truncated) s += ` · ${c.truncated} more left out (Settings -> Market monitor -> Max stocks watched)`;
      return s;
    },
    get thresholdText() {
      const t = this.data && this.data.thresholds;
      if (!t) return "";
      return `A spike = a move of ${t.spike_pct}% or more within ${t.spike_window_minutes} min on at least ${t.volume_ratio}x ` +
        `the usual volume. Market alerts: the S&P 500 and friends moving ${t.market_move_pct}% within 15 minutes, and ` +
        `every ${t.market_day_step_pct}% up or down on the day. Change these in Settings -> Market monitor.`;
    },
  }));
});
