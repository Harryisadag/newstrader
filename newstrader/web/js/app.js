// Root app: tabs, header status, toasts, websocket wiring.
document.addEventListener("alpine:init", () => {
  const TABS = [
    { id: "live", label: "Live" },
    { id: "signals", label: "Signals" },
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

    async init() {
      window.addEventListener("nt:connection", (e) => { this.store.connected = e.detail.connected; if (e.detail.connected) this.refresh(); });
      window.addEventListener("nt:heartbeat", (e) => { this.store.status = { ...this.store.status, ...e.detail }; });
      window.addEventListener("nt:status", (e) => this.upsertComponent(e.detail));
      window.addEventListener("nt:status_removed", (e) => {
        this.store.status.components = this.store.status.components.filter((c) => c.component !== e.detail.component);
      });
      window.addEventListener("nt:mode", (e) => { this.store.status.mode = e.detail.mode; });
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
      if (!s.keys || !s.keys.alpaca_paper) return { label: "NO KEYS", cls: "warn", tip: "Add your Alpaca paper keys in Settings -> API Keys." };
      if (!s.auto_trade) return { label: "MONITOR ONLY", cls: "warn", tip: "Auto-trade is off (Settings -> Trading)." };
      return { label: "ACTIVE", cls: "good", tip: "Auto-trading is on." };
    },

    get spendPct() {
      const cap = Number(this.s.spend_cap || 0);
      if (!cap) return 0;
      return Math.min(100, (100 * Number(this.s.spend_today || 0)) / cap);
    },

    get problemCount() {
      return (this.s.components || []).filter((c) => c.level === "error").length;
    },
  }));
});
