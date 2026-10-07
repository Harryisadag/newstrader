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
        F("trading.max_chase_pct", "Don't chase moves bigger than", "number", { min: 0, max: 50, step: 0.5, suffix: "%", help: "If the price already moved this much in the signal's direction since the news came out, the signal goes to manual review instead of being bought - the move may already be over. 0 = off." }),
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
        F("ai.engine", "AI engine", "select", { optionsFrom: "engines", help: "Local = free machine learning on this computer (FinBERT reads the wording; a model you train on price history checks it). Claude = reads context better but costs money per story and needs an Anthropic key." }),
        F("ml.sentiment_model", "Sentiment model", "select", { local: true, options: { finbert: "FinBERT (recommended - one-time ~110 MB download)", lexicon: "Built-in word list (no download, crude - confidence capped at 79)" } }),
        F("ml.use_trained_model", "Use the price-trained model", "select", { local: true, options: { auto: "Auto - only if it passed its test", always: "Always (even if it didn't pass)", never: "Never - sentiment only" }, help: "Train it in the Backtest tab -> Local ML model. Its confidence is a tested probability, so it is usually lower than sentiment-only confidence." }),
        F("ml.event_rules", "Recognise news events", "toggle", { local: true, help: "Spots concrete events - earnings beats and misses, guidance changes, buyouts (good for the company being bought), FDA decisions, analyst upgrades and downgrades, share offerings, buybacks, lawsuits - and works out which company each one is good or bad for. Recaps of moves that already happened, opinion pieces and unconfirmed reports get lower confidence." }),
        F("ml.country_etfs", "International news -> country ETFs", "toggle", { local: true, help: "News about a country's economy (e.g. 'Bank of Japan raises rates') can create signals for US-listed country funds such as EWJ (Japan) or FXI (China). These always go to manual review." }),
        F("ml.sentiment_only_trading", "Sentiment-only signals can auto-buy", "select", { local: true, options: { auto: "Yes - unless a trained price model found wording doesn't predict moves", always: "Yes, always", review: "No - manual review only" }, help: "When no price model is in use, confidence is FinBERT's reading of the wording - not a tested prediction of the price. While LIVE (real-money) trading is on, these signals always go to manual review." }),
        F("ml.train_days", "Training history", "number", { local: true, min: 30, max: 730, suffix: "days", help: "More history = more examples to learn from (and a longer first download)." }),
        F("ml.train_horizon_minutes", "Measure each stock's move over", "select", { local: true, numeric: true, options: { 30: "30 minutes", 60: "1 hour", 120: "2 hours" } }),
        F("ml.train_min_move_pct", "Smallest move that counts", "number", { local: true, min: 0, max: 5, step: 0.05, suffix: "% vs S&P 500", help: "Headlines after which the stock moved less than this (compared with the market) are left out of training." }),
        F("ml.train_max_articles", "Max headlines to download", "number", { local: true, min: 500, max: 100000, step: 500 }),
        F("ai.model", "Claude model", "select", { claude: true, optionsFrom: "claude_models" }),
        F("ai.effort", "Thinking effort", "select", { claude: true, options: { low: "Low (fast, cheapest)", medium: "Medium", high: "High (slowest, most thorough)" }, help: "Sonnet/Opus only. Higher = more careful but slower and more expensive." }),
        F("ai.daily_spend_cap_usd", "Daily Claude spend cap", "number", { claude: true, min: 0, step: 1, prefix: "$", help: "When today's estimated API cost reaches this, Claude calls stop until tomorrow." }),
        F("ai.max_calls_per_minute", "Max Claude calls per minute", "number", { claude: true, min: 1, max: 300 }),
        F("ai.analyze_keyword_only", "Analyse macro news with no company named", "toggle", { claude: true, help: "e.g. 'Fed cuts rates' - Claude may pick an affected stock or ETF." }),
        F("ai.use_refusal_fallback", "Refusal fallback (Sonnet 5.5)", "toggle", { claude: true, help: "If Claude declines a request for safety reasons, Anthropic retries it on a fallback model automatically." }),
        F("ai.max_concurrent_calls", "Stories analysed at the same time", "number", { min: 1, max: 10 }),
        F("ai.story_dedupe_minutes", "Same-story window", "number", { min: 0, max: 240, suffix: "minutes", help: "The same headline from several sources within this window is analysed only once." }),
        F("ai.story_similarity", "Same-story similarity", "number", { min: 0.3, max: 1, step: 0.05, help: "How similar two headlines must be to count as the same story (0.3 = loose, 1 = identical)." }),
        F("ai.signal_dedupe_minutes", "Same-signal window", "number", { min: 0, max: 240, suffix: "minutes", help: "Same ticker + direction within this window counts as one signal (traded at most once)." }),
        F("ai.max_signals_per_item", "Max stocks per story", "number", { min: 1, max: 5 }),
      ],
    },
    {
      id: "transcription", label: "Transcription", fields: [
        F("transcription.enabled", "Transcribe live streams", "toggle"),
        F("transcription.max_concurrent_streams", "Max streams at once", "number", { min: 1, max: 12 }),
        F("transcription.model", "Whisper model", "select", { optionsFrom: "whisper_models", help: "large-v3 is the most accurate (~3 GB download on first use). large-v3-turbo is ~4x faster with nearly the same accuracy (recommended on a Mac)." }),
        F("transcription.device", "Device", "select", { options: { cuda: "NVIDIA GPU (CUDA)", mlx: "Apple GPU (Mac with M-series chip)", cpu: "CPU (slow)", auto: "Auto (best available)" } }),
        F("transcription.compute_type", "Precision", "select", { options: { float16: "float16 (recommended for RTX 50)", int8_float16: "int8_float16", int8: "int8 (CPU / Mac)", float32: "float32" }, help: "NVIDIA GPUs only. The Apple GPU ignores this; CPUs always use int8." }),
        F("transcription.language", "Language", "select", { optionsFrom: "languages", help: "The language most streams speak. Each stream can override this (Settings -> News sources), and can translate to English." }),
        F("transcription.beam_size", "Beam size", "number", { min: 1, max: 10, help: "Higher = slightly more accurate, slower." }),
        F("transcription.chunk_seconds", "Audio chunk length", "number", { min: 3, max: 30, suffix: "seconds" }),
        F("transcription.vad_min_silence_ms", "Silence that splits speech", "number", { min: 100, max: 3000, suffix: "ms" }),
        F("transcription.analysis_window_seconds", "Transcript context analysed", "number", { min: 15, max: 300, suffix: "seconds" }),
        F("transcription.analysis_debounce_seconds", "Wait for sentence to finish", "number", { min: 0, max: 60, suffix: "seconds" }),
        F("transcription.cookies_from_browser", "YouTube cookies from browser", "select", { options: { "": "None", chrome: "Chrome", edge: "Edge", firefox: "Firefox", brave: "Brave", safari: "Safari (Mac)" }, help: "Only if YouTube says 'Sign in to confirm you're not a bot'. On a Mac, Chrome asks for Keychain access and Safari needs Full Disk Access for NewsTrader." }),
      ],
    },
    {
      id: "alerts", label: "Alerts", fields: [
        F("alerts.desktop_enabled", "Desktop pop-ups (Windows / Mac)", "toggle"),
        F("alerts.discord_enabled", "Discord", "toggle", { help: "Needs a webhook URL in API Keys." }),
        F("alerts.on_trade_placed", "Trade placed", "toggle"),
        F("alerts.on_trade_filled", "Trade filled / closed", "toggle"),
        F("alerts.on_manual_review", "Manual review signals", "toggle"),
        F("alerts.on_error", "Errors", "toggle"),
        F("alerts.on_daily_loss_limit", "Daily loss limit hit", "toggle"),
        F("alerts.on_spend_cap", "Claude spend cap hit", "toggle"),
        F("alerts.on_kill_switch", "Kill switch used", "toggle"),
        F("alerts.on_market_spike", "Sudden price spikes", "toggle", { help: "A watched stock jumps or drops fast on unusual volume (see Settings -> Market monitor)." }),
        F("alerts.on_market_move", "Big market-wide moves", "toggle", { help: "The S&P 500, Nasdaq, small caps or Dow (via their ETFs) move sharply." }),
      ],
    },
    {
      id: "market", label: "Market monitor", fields: [
        F("market.enabled", "Watch the market", "toggle", { help: "Checks prices about once a minute for sudden spikes and big market-wide moves, and shows today's top movers on the Market tab." }),
        F("market.scan_seconds", "Check every", "number", { min: 15, max: 600, suffix: "seconds" }),
        F("market.spike_window_minutes", "Spike = a move within", "select", { numeric: true, options: { 1: "1 minute", 5: "5 minutes", 15: "15 minutes" } }),
        F("market.spike_pct", "Spike size", "number", { min: 0.5, max: 50, step: 0.5, suffix: "%", help: "How much the price must move within that time." }),
        F("market.volume_ratio", "On volume at least", "number", { min: 1, max: 100, step: 0.5, suffix: "x normal", help: "Compared with the stock's usual volume for that many minutes. Free IEX data only sees part of all trading, so volume is a rough guide." }),
        F("market.min_price", "Ignore stocks under", "number", { min: 0, step: 0.5, prefix: "$" }),
        F("market.watch_positions", "Watch stocks I hold", "toggle"),
        F("market.watch_signals_minutes", "Watch stocks with signals in the last", "number", { min: 0, max: 1440, suffix: "minutes", help: "0 = don't." }),
        F("market.watch_movers", "Watch today's top movers", "toggle", { help: "Alpaca's list of the day's biggest gainers, losers and most active stocks." }),
        F("market.movers_top", "Top movers to include", "number", { min: 5, max: 50 }),
        F("market.watchlist", "Always watch", "list", { placeholder: "e.g. AAPL, NVDA, TSLA", help: "Comma-separated tickers." }),
        F("market.max_symbols", "Max stocks watched", "number", { min: 10, max: 400 }),
        F("market.index_symbols", "Market gauges", "list", { help: "ETFs that track the whole market: SPY = S&P 500, QQQ = Nasdaq 100, IWM = small caps, DIA = Dow." }),
        F("market.world_symbols", "World markets", "list", { help: "US-listed country ETFs, e.g. EWJ = Japan, FXI = China, EWG = Germany, EWU = UK, INDA = India, EWZ = Brazil, EZU = Eurozone, EWY = South Korea." }),
        F("market.market_move_pct", "Market move alert", "number", { min: 0.2, max: 10, step: 0.1, suffix: "% in 15 min" }),
        F("market.market_day_step_pct", "...and every", "number", { min: 0.5, max: 10, step: 0.5, suffix: "% on the day" }),
        F("market.extended_hours", "Also before/after market hours", "toggle", { help: "4am-9:30am and 4pm-8pm ET. Fewer trades then, so more false alarms." }),
        F("market.lookup_news", "Look up the news behind a spike", "toggle"),
        F("market.alert_cooldown_minutes", "Don't re-alert the same stock for", "number", { min: 1, max: 1440, suffix: "minutes" }),
        F("market.max_alerts_per_scan", "Max alerts per check", "number", { min: 1, max: 20 }),
      ],
    },
    {
      id: "display", label: "Display", fields: [
        F("ui.time_zone", "Show times in", "select", { options: { local: "This computer's time zone", market: "New York time (ET, the US market)", utc: "UTC" } }),
        F("ui.check_updates", "Tell me when a new version is out", "toggle", { help: "Checks GitHub once a day and shows a banner with a download link. Nothing is installed by itself." }),
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
        const w = r.desktop ? "Desktop pop-up: shown ✓" : "Desktop pop-up: not available on this system";
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
    visible(f) {
      const engine = this.get("ai.engine");
      if (f.local) return engine === "local";
      if (f.claude) return engine === "claude";
      return true;
    },
    pick(f, value) { if (f.numeric) this.num(f.key, value); else this.set(f.key, value); },

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
