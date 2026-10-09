// Root app: tabs, header status, toasts, websocket wiring.
document.addEventListener("alpine:init", () => {
  const TABS = [
    { id: "live", label: "Live" },
    { id: "signals", label: "Signals" },
    { id: "market", label: "Market" },
    { id: "portfolio", label: "Portfolio" },
    { id: "trades", label: "Trade Log" },
    { id: "performance", label: "Performance" },
    { id: "backtest", label: "Backtest" },
    { id: "settings", label: "Settings" },
    { id: "logs", label: "Logs" },
  ];

  let savedTab = "live";
  try { savedTab = localStorage.getItem("nt_tab") || "live"; } catch (e) { /* storage blocked */ }
  if (!TABS.some((t) => t.id === savedTab)) savedTab = "live";

  Alpine.store("nt", {
    tabs: TABS,
    tab: savedTab,
    status: { mode: "paper", kill_switch: { engaged: false }, components: [], keys: {}, spend_today: 0, spend_cap: 0 },
    connected: false,
    toasts: [],
    _toastId: 0,
    setTab(id) {
      this.tab = id;
      try { localStorage.setItem("nt_tab", id); } catch (e) { /* ignore */ }
      window.dispatchEvent(new CustomEvent("nt:tab", { detail: id }));
    },
    toast(kind, title, message = "", timeout = 6000) {
      const id = ++this._toastId;
      this.toasts.push({ id, kind, title, message });
      if (this.toasts.length > 6) this.toasts.shift();
      if (timeout) setTimeout(() => this.dismiss(id), timeout);
    },
    dismiss(id) {
      this.toasts = this.toasts.filter((t) => t.id !== id);
    },
    error(e, title = "Something went wrong") {
      const details = e && e.details && e.details.length ? "\n• " + e.details.join("\n• ") : "";
      this.toast("error", title, ((e && e.message) || String(e)) + details, 10000);
    },
  });

  Alpine.data("app", () => ({
    fmt: NT.fmt,
    get s() { return Alpine.store("nt").status; },
    get store() { return Alpine.store("nt"); },
    dismissedUpdate: (() => { try { return localStorage.getItem("nt_dismissed_update") || ""; } catch (e) { return ""; } })(),

    dismissUpdate() {
      this.dismissedUpdate = (this.s.update && this.s.update.latest) || "";
      try { localStorage.setItem("nt_dismissed_update", this.dismissedUpdate); } catch (e) { /* private window */ }
    },

    // "Update now" (the ready-made app only): download, check, swap in and restart
    get install() { return this.s.update_install || {}; },
    get canUpdateNow() { return !!(this.s.update && !this.s.update.source_mode && this.install.supported); },
    get updating() { return !!this.install.busy; },
    get updateActive() { return (!!this.install.phase && this.install.phase !== "idle") || !!this.install.message; },
    get updatePct() { const i = this.install; return i.total ? Math.min(100, Math.round((100 * i.done) / i.total)) : 0; },
    get updateText() {
      const i = this.install;
      if (i.phase === "downloading" && i.total) return `${i.message} ${this.updatePct}% of ${Math.round(i.total / 1e6)} MB`;
      return i.message || "";
    },
    async updateNow() {
      const u = this.s.update || {};
      const size = u.size ? ` (about ${Math.round(u.size / 1e6)} MB)` : "";
      const ok = confirm(`Update to NewsTrader ${u.latest} now?\n\n`
        + `It downloads the new version${size} and checks it's intact. Then NewsTrader closes and opens again by itself, `
        + "which takes about a minute. No trades are placed while it restarts.\n\n"
        + "Your settings, keys, history and the kill switch stay as they are. If live trading is on, NewsTrader comes "
        + "back in paper mode, like after any restart.");
      if (!ok) return;
      try {
        this.store.status = { ...this.store.status, update_install: await NT.api.post("/updates/install") };
      } catch (e) {
        this.store.error(e, "The update didn't start");
      }
    },
    async cancelUpdate() {
      try { this.store.status = { ...this.store.status, update_install: await NT.api.post("/updates/cancel") }; }
      catch (e) { this.store.error(e); }
    },

    async init() {
      window.addEventListener("nt:connection", (e) => { this.store.connected = e.detail.connected; if (e.detail.connected) this.refresh(); });
      window.addEventListener("nt:heartbeat", (e) => { this.store.status = { ...this.store.status, ...e.detail }; });
      window.addEventListener("nt:status", (e) => this.upsertComponent(e.detail));
      window.addEventListener("nt:status_removed", (e) => {
        this.store.status.components = this.store.status.components.filter((c) => c.component !== e.detail.component);
      });
      window.addEventListener("nt:mode", (e) => { this.store.status.mode = e.detail.mode; });
      window.addEventListener("nt:pro_ai", (e) => { this.store.status = { ...this.store.status, pro_ai: e.detail }; });
      window.addEventListener("nt:update_install", (e) => { this.store.status = { ...this.store.status, update_install: e.detail }; });
      window.addEventListener("nt:kill_switch", (e) => { this.store.status.kill_switch = e.detail; });
      window.addEventListener("nt:toast", (e) => this.store.toast(e.detail.kind || "info", e.detail.title, e.detail.message || ""));
      NT.connect();
      await this.refresh();
    },

    async refresh() {
      try {
        this.store.status = await NT.api.get("/status");
      } catch (e) {
        if (e.status === 401) this.store.toast("error", "Session expired", "Close and reopen NewsTrader.", 0);
      }
    },

    upsertComponent(c) {
      const list = this.store.status.components || [];
      const i = list.findIndex((x) => x.component === c.component);
      if (i >= 0) list.splice(i, 1, c); else list.push(c);
      this.store.status.components = [...list];
    },

    get tradingState() {
      const s = this.s;
      if (s.kill_switch && s.kill_switch.engaged) return { label: "KILLED", cls: "bad", tip: "Kill switch engaged - no trading until you re-arm it." };
      if (s.halted_today) return { label: "HALTED", cls: "bad", tip: "Daily loss limit hit - trading resumes next trading day." };
      if (!s.broker_connected) {
        if (!s.keys || !s.keys.alpaca_paper) return { label: "NO KEYS", cls: "warn", tip: "Add your Alpaca paper keys in Settings -> API Keys." };
        return { label: "OFFLINE", cls: "bad", tip: s.broker_error || "Not connected to Alpaca - see Logs." };
      }
      if (!s.auto_trade) return { label: "MONITOR ONLY", cls: "warn", tip: "Auto-trade is off (Settings -> Trading)." };
      return { label: "ACTIVE", cls: "good", tip: "Auto-trading is on." };
    },

    // the small Pro AI light in the header (shown while Pro AI is turned on)
    get proLight() {
      const p = this.s.pro_ai || {};
      const dot = { ready: "ok", starting: "starting", downloading: "starting", error: "error", not_set_up: "warn" }[p.state] || "off";
      let label = { starting: "starting", error: "problem", not_set_up: "not set up", off: "stopped" }[p.state] || p.state || "";
      if (p.state === "ready") label = p.mode === "judge" ? "judging" : "watching";
      if (p.state === "downloading" && p.progress && p.progress.total) {
        label = `downloading ${Math.round((100 * p.progress.done) / p.progress.total)}%`;
      }
      return { dot, label };
    },

    get spendPct() {
      const cap = Number(this.s.spend_cap || 0);
      if (!cap) return 0;
      return Math.min(100, (100 * Number(this.s.spend_today || 0)) / cap);
    },

    goto(tab, section = null) {
      this.store.setTab(tab);
      if (section) window.dispatchEvent(new CustomEvent("nt:settings-section", { detail: section }));
    },

    get problemCount() {
      return (this.s.components || []).filter((c) => c.level === "error").length;
    },
  }));
});
