// Settings tab: every option, saved to config.json. Also Sources and API Keys.
document.addEventListener("alpine:init", () => {
  const F = (key, label, type, extra = {}) => ({ key, label, type, ...extra });

  const SECTIONS = [
    {
      id: "trading", label: "Trading", fields: [
        F("trading.auto_trade", "Auto-trade", "toggle", { help: "When off, signals are still recorded and alerted, but no orders are placed." }),
        F("trading.buy_threshold", "Buy threshold", "number", { min: 1, max: 100, suffix: "confidence", help: "Bullish signals at or above this confidence are bought automatically." }),
        F("trading.review_threshold", "Manual review threshold", "number", { min: 0, max: 100, suffix: "confidence", help: "Signals between this and the buy threshold send a 'manual review' alert instead." }),
        F("trading.stop_loss_pct", "Stop-loss", "number", { min: 0.1, max: 50, step: 0.1, suffix: "%", help: "Every buy is a bracket order with this stop-loss below the entry price." }),
        F("trading.take_profit_pct", "Take-profit", "number", { min: 0.1, max: 200, step: 0.1, suffix: "%", help: "...and this take-profit above the entry price." }),
        F("trading.sell_on_bearish", "Sell on bearish signal", "toggle", { help: "If a bearish signal clears the buy threshold and you hold the stock, close the position." }),
        F("trading.allow_shorting", "Allow short selling", "toggle", { help: "Bearish signals open a short (bracket order) when you don't hold the stock. Off by default." }),
        F("trading.data_feed", "Market data feed", "select", { options: { iex: "IEX (free)", sip: "SIP (paid Alpaca data plan)" } }),
      ],
    },
    {
      id: "risk", label: "Risk controls", fields: [
        F("risk.max_dollars_per_trade", "Max $ per trade", "number", { min: 1, step: 50, prefix: "$" }),
        F("risk.max_pct_per_stock", "Max % of account in one stock", "number", { min: 0.1, max: 100, step: 0.5, suffix: "%" }),
        F("risk.max_open_positions", "Max open positions", "number", { min: 1, max: 200 }),
        F("risk.daily_loss_limit_usd", "Daily loss limit", "number", { min: 1, step: 50, prefix: "$", help: "When today's loss reaches this, the bot stops trading until the next trading day." }),
        F("risk.ticker_cooldown_minutes", "Cooldown per ticker", "number", { min: 0, max: 10080, suffix: "minutes", help: "Don't trade the same stock again within this many minutes." }),
        F("risk.market_hours_only", "Market hours only", "toggle", { help: "Only trade 9:30am-4:00pm ET. If off, orders sent while the market is closed are queued by Alpaca for the next open (bracket orders can't trade pre/after-market)." }),
        F("risk.min_share_price", "Minimum share price", "number", { min: 0, step: 0.5, prefix: "$", help: "Skip penny stocks below this price." }),
        F("risk.blacklist", "Blacklist (never trade)", "list", { placeholder: "e.g. GME, AMC", help: "Comma-separated tickers." }),
        F("risk.whitelist", "Whitelist (only trade these)", "list", { placeholder: "empty = every stock allowed", help: "Leave empty to allow everything." }),
      ],
    },
    {
      id: "ai", label: "AI engine", fields: [
        F("ai.model", "Claude model", "select", { optionsFrom: "claude_models" }),
        F("ai.effort", "Thinking effort", "select", { options: { low: "Low (fast, cheapest)", medium: "Medium", high: "High (slowest, most thorough)" }, help: "Sonnet/Opus only. Higher = more careful but slower and more expensive." }),
        F("ai.daily_spend_cap_usd", "Daily Claude spend cap", "number", { min: 0, step: 1, prefix: "$", help: "When today's estimated API cost reaches this, Claude calls stop until tomorrow." }),
        F("ai.max_calls_per_minute", "Max Claude calls per minute", "number", { min: 1, max: 300 }),
        F("ai.max_concurrent_calls", "Max simultaneous Claude calls", "number", { min: 1, max: 10 }),
        F("ai.story_dedupe_minutes", "Same-story window", "number", { min: 0, max: 240, suffix: "minutes", help: "The same headline from several sources within this window is analysed only once." }),
        F("ai.story_similarity", "Same-story similarity", "number", { min: 0.3, max: 1, step: 0.05, help: "How similar two headlines must be to count as the same story (0.3 = loose, 1 = identical)." }),
        F("ai.signal_dedupe_minutes", "Same-signal window", "number", { min: 0, max: 240, suffix: "minutes", help: "Same ticker + direction within this window counts as one signal (traded at most once)." }),
        F("ai.max_signals_per_item", "Max stocks per story", "number", { min: 1, max: 5 }),
        F("ai.analyze_keyword_only", "Analyse macro news with no company named", "toggle", { help: "e.g. 'Fed cuts rates' - Claude may pick an affected stock or ETF." }),
        F("ai.use_refusal_fallback", "Refusal fallback (Sonnet 5.5)", "toggle", { help: "If Claude declines a request for safety reasons, Anthropic retries it on a fallback model automatically." }),
      ],
    },
    {
      id: "transcription", label: "Transcription", fields: [
        F("transcription.enabled", "Transcribe live streams", "toggle"),
        F("transcription.max_concurrent_streams", "Max streams at once", "number", { min: 1, max: 12 }),
        F("transcription.model", "Whisper model", "select", { optionsFrom: "whisper_models", help: "large-v3 is the most accurate (~3 GB download on first use). large-v3-turbo is faster." }),
        F("transcription.device", "Device", "select", { options: { cuda: "GPU (CUDA)", cpu: "CPU (slow)", auto: "Auto" } }),
        F("transcription.compute_type", "Precision", "select", { options: { float16: "float16 (recommended for RTX 50)", int8_float16: "int8_float16", int8: "int8 (CPU)", float32: "float32" } }),
        F("transcription.language", "Language", "text", { help: "'en' for English, or 'auto' to detect." }),
        F("transcription.beam_size", "Beam size", "number", { min: 1, max: 10, help: "Higher = slightly more accurate, slower." }),
        F("transcription.chunk_seconds", "Audio chunk length", "number", { min: 3, max: 30, suffix: "seconds" }),
        F("transcription.vad_min_silence_ms", "Silence that splits speech", "number", { min: 100, max: 3000, suffix: "ms" }),
        F("transcription.analysis_window_seconds", "Transcript context sent to Claude", "number", { min: 15, max: 300, suffix: "seconds" }),
        F("transcription.analysis_debounce_seconds", "Wait for sentence to finish", "number", { min: 0, max: 60, suffix: "seconds" }),
        F("transcription.cookies_from_browser", "YouTube cookies from browser", "select", { options: { "": "None", chrome: "Chrome", edge: "Edge", firefox: "Firefox", brave: "Brave" }, help: "Only if YouTube says 'Sign in to confirm you're not a bot'." }),
      ],
    },
    {
      id: "alerts", label: "Alerts", fields: [
        F("alerts.desktop_enabled", "Windows desktop popups", "toggle"),
        F("alerts.discord_enabled", "Discord", "toggle", { help: "Needs a webhook URL in API Keys." }),
        F("alerts.on_trade_placed", "Trade placed", "toggle"),
        F("alerts.on_trade_filled", "Trade filled / closed", "toggle"),
        F("alerts.on_manual_review", "Manual review signals", "toggle"),
        F("alerts.on_error", "Errors", "toggle"),
        F("alerts.on_daily_loss_limit", "Daily loss limit hit", "toggle"),
        F("alerts.on_spend_cap", "Claude spend cap hit", "toggle"),
        F("alerts.on_kill_switch", "Kill switch used", "toggle"),
      ],
    },
  ];

  const getPath = (obj, path) => path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
  const setPath = (obj, path, value) => {
    const parts = path.split(".");
    const last = parts.pop();
    const target = parts.reduce((o, k) => (o[k] = o[k] || {}), obj);
    target[last] = value;
  };

  Alpine.data("settingsTab", () => ({
    sections: SECTIONS,
    section: "trading",
    form: null,
    original: "",
    meta: {},
    errors: [],
    saving: false,
    alertTesting: false,
    alertResult: "",

    async testAlert() {
      this.alertTesting = true; this.alertResult = "";
      try {
        const r = await NT.api.post("/alerts/test");
        const d = r.discord === true ? "Discord: sent ✓" : (r.discord ? "Discord: " + r.discord : "Discord: not set up");
        const w = r.desktop ? "Desktop pop-up: shown ✓" : "Desktop pop-up: not available (Windows only)";
        this.alertResult = `${w} · ${d}`;
      } catch (e) { Alpine.store("nt").error(e, "Test alert failed"); }
      finally { this.alertTesting = false; }
    },

    async init() {
      window.addEventListener("nt:settings-section", (e) => { this.section = e.detail; });
      await this.load();
      window.addEventListener("nt:settings_changed", () => { if (!this.dirty) this.load(); });
    },

    async load() {
      try {
        const res = await NT.api.get("/settings");
        this.meta = res.meta;
        const form = res.settings;
        for (const sec of SECTIONS) for (const f of sec.fields) {
          if (f.type === "list") setPath(form, f.key, (getPath(form, f.key) || []).join(", "));
        }
        this.form = form;
        this.original = JSON.stringify(form);
        this.errors = [];
      } catch (e) { Alpine.store("nt").error(e, "Couldn't load settings"); }
    },

    get dirty() { return this.form !== null && JSON.stringify(this.form) !== this.original; },
    get(path) { return this.form ? getPath(this.form, path) : undefined; },
    set(path, value) { setPath(this.form, path, value); },
    num(path, value) { const n = parseFloat(value); this.set(path, isNaN(n) ? value : n); },
    options(f) {
      if (f.options) return f.options;
      const src = this.meta[f.optionsFrom];
      if (Array.isArray(src)) return Object.fromEntries(src.map((x) => [x, x]));
      const opts = { ...(src || {}) };
      const cur = this.get(f.key);
      if (cur !== undefined && cur !== null && !(cur in opts)) opts[cur] = cur + " (custom)";
      return opts;
    },
    get currentSection() { return SECTIONS.find((s) => s.id === this.section); },

    async save() {
      this.saving = true;
      this.errors = [];
      try {
        const payload = JSON.parse(JSON.stringify(this.form));
        await NT.api.put("/settings", payload);
        await this.load();
        Alpine.store("nt").toast("success", "Settings saved");
      } catch (e) {
        this.errors = e.details && e.details.length ? e.details : [e.message];
        Alpine.store("nt").toast("error", "Settings not saved", "Fix the highlighted problems.");
      } finally { this.saving = false; }
    },

    discard() { this.form = JSON.parse(this.original); this.errors = []; },

    async resetDefaults() {
      if (!confirm("Reset ALL settings (except your news sources) to defaults?")) return;
      try { await NT.api.post("/settings/reset"); await this.load(); Alpine.store("nt").toast("success", "Defaults restored"); }
      catch (e) { Alpine.store("nt").error(e); }
    },
  }));

  // ---- Sources manager (used in Settings -> Sources) ----
  Alpine.data("sourcesManager", () => ({
    sources: [],
    types: {},
    filter: "all",
    adding: false,
    draft: { type: "rss", name: "", url: "", speaker: "", poll_seconds: 60 },
    fmt: NT.fmt,

    async init() {
      await this.load();
      window.addEventListener("nt:sources_changed", () => this.load());
      window.addEventListener("nt:status", (e) => {
        const c = e.detail;
        if (!c.component.startsWith("source:")) return;
        const id = c.component.slice(7);
        const s = this.sources.find((x) => x.id === id);
        if (s) s.health = c;
      });
    },
    async load() {
      try { const r = await NT.api.get("/sources"); this.sources = r.sources; this.types = r.types; }
      catch (e) { Alpine.store("nt").error(e, "Couldn't load sources"); }
    },
    get visible() { return this.filter === "all" ? this.sources : this.sources.filter((s) => s.type === this.filter); },
    typeLabel(t) { return { stream: "Live stream", rss: "RSS", social_rss: "Social", x_account: "X (API)", alpaca_news: "Alpaca news" }[t] || t; },
    async toggle(src) {
      try { await NT.api.post(`/sources/${src.id}/toggle`, { enabled: !src.enabled }); src.enabled = !src.enabled; }
      catch (e) { Alpine.store("nt").error(e); }
    },
    async remove(src) {
      if (!confirm(`Remove "${src.name}"?`)) return;
      try { await NT.api.del(`/sources/${src.id}`); await this.load(); }
      catch (e) { Alpine.store("nt").error(e); }
    },
    async add() {
      try {
        await NT.api.post("/sources", this.draft);
        Alpine.store("nt").toast("success", "Source added", this.draft.name);
        this.draft = { type: this.draft.type, name: "", url: "", speaker: "", poll_seconds: 60 };
        this.adding = false;
        await this.load();
      } catch (e) { Alpine.store("nt").error(e, "Couldn't add source"); }
    },
    urlPlaceholder() {
      return { stream: "https://www.youtube.com/@channel/live  (or any stream URL)", rss: "https://example.com/feed.xml",
        social_rss: "RSS feed URL for the account", x_account: "X username, e.g. elonmusk", alpaca_news: "(no URL needed)" }[this.draft.type];
    },
  }));

  // ---- API keys (Settings -> API Keys) ----
  Alpine.data("keysManager", () => ({
    keys: [],
    envFile: "",
    edits: {},
    saving: false,
    async init() { await this.load(); },
    async load() {
      try { const r = await NT.api.get("/keys"); this.keys = r.keys; this.envFile = r.env_file; this.edits = {}; }
      catch (e) { Alpine.store("nt").error(e, "Couldn't load keys"); }
    },
    get changed() { return Object.keys(this.edits).filter((k) => this.edits[k] !== undefined && this.edits[k] !== ""); },
    isLive(name) { return name.startsWith("ALPACA_LIVE"); },
    async save(clearName = null) {
      const values = {};
      if (clearName) values[clearName] = "";
      else for (const k of this.changed) values[k] = this.edits[k];
      if (!Object.keys(values).length) return;
      this.saving = true;
      try {
        await NT.api.put("/keys", values);
        await this.load();
        Alpine.store("nt").toast("success", clearName ? "Key removed" : "Keys saved", "Saved to " + this.envFile);
        window.dispatchEvent(new CustomEvent("nt:refresh-status"));
      } catch (e) { Alpine.store("nt").error(e, "Keys not saved"); }
      finally { this.saving = false; }
    },
  }));
});
