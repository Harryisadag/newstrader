// Market tab: the whole market (S&P 500, Nasdaq... via their ETFs), world markets, sudden spikes, today's top
// movers and the stocks being watched. Live updates: "market_event" (one spike or move) and "market" (after each check).
document.addEventListener("alpine:init", () => {
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

    init() {
      window.addEventListener("nt:tab", (e) => { if (e.detail === "market") this.load(); });
      window.addEventListener("nt:market_event", (e) => this.addEvent(e.detail));
      window.addEventListener("nt:market", (e) => this.onSummary(e.detail));
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
    },

    async scanNow() {
      this.scanning = true;
      try { await NT.api.post("/market/scan"); await this.load(true); }
      catch (e) { Alpine.store("nt").error(e, "Couldn't check the market"); }
      finally { this.scanning = false; }
    },

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
