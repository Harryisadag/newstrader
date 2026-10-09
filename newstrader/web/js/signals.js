// Signals tab: every AI signal (and every chart signal), the manual-review queue, and a "test the AI" box.
document.addEventListener("alpine:init", () => {
  const MAX_SIGNALS = 500;  // rows kept in the list (the newest)
  Alpine.data("signalsTab", () => ({
    fmt: NT.fmt,
    signals: [],
    reviews: [],  // the manual-review queue: loaded on its own, so busy watch-only rows never push one out
    // watch-only calls (Pro AI's and the chart's) are hidden at first: they are never traded and can be many
    filters: { ticker: "", direction: "", action: "", engine: "main", min_confidence: 0 },
    expanded: null,
    detail: {},
    stats: null,
    busy: {},
    testText: "",
    testing: false,
    testResult: null,

    async init() {
      window.addEventListener("nt:signal", (e) => this.upsert(e.detail));
      window.addEventListener("nt:signal_updated", () => {
        if (Alpine.store("nt").tab === "signals") { this.load(true); this.loadReviews(); }
      });
      window.addEventListener("nt:tab", (e) => { if (e.detail === "signals") { this.load(); this.loadReviews(); this.loadStats(); } });
      await Promise.all([this.load(), this.loadReviews()]);
      this.loadStats();
    },

    async load(quiet = false) {
      const f = this.filters;
      const q = `limit=${MAX_SIGNALS}&ticker=${encodeURIComponent(f.ticker)}&direction=${f.direction}&action=${f.action}&engine=${f.engine}&min_confidence=${f.min_confidence || 0}`;
      try { this.signals = (await NT.api.get("/signals?" + q)).signals; }
      catch (e) { if (!quiet) Alpine.store("nt").error(e, "Couldn't load signals"); }
    },
    async loadStats() { try { this.stats = await NT.api.get("/ai/stats"); } catch (e) { /* ignore */ } },
    async loadReviews() {
      try { this.reviews = (await NT.api.get("/signals?review_only=true&limit=1000")).signals; } catch (e) { /* ignore */ }
    },

    upsert(s) {
      if (s.merged_into) return;
      if (typeof s.sources_seen === "string") { try { s.sources_seen = JSON.parse(s.sources_seen); } catch (e) { s.sources_seen = []; } }
      const r = this.reviews.findIndex((x) => x.id === s.id);
      if (this.pending(s)) { if (r >= 0) this.reviews.splice(r, 1, s); else this.reviews.unshift(s); }
      else if (r >= 0) this.reviews.splice(r, 1);
      const f = this.filters;
      if ((f.engine === "main" && s.action === "watch") || (f.engine === "pro" && s.engine !== "pro") ||
          (f.engine === "chart" && s.engine !== "chart") || (f.engine === "news" && s.engine === "chart")) return;
      const i = this.signals.findIndex((x) => x.id === s.id);
      if (i >= 0) this.signals.splice(i, 1, s); else this.signals.unshift(s);
      if (this.signals.length > MAX_SIGNALS) this.signals.splice(MAX_SIGNALS);
      this.loadStats();
    },

    pending(s) { return s.action === "review" && (!s.review_status || s.review_status === "pending"); },
    get reviewQueue() { return this.reviews.filter((s) => this.pending(s)); },
    get today() {
      const d = new Date().toDateString();
      // watch-only calls (Pro AI's and the chart's) aren't counted - they are never traded
      const t = this.signals.filter((s) => new Date(s.created_at).toDateString() === d && s.action !== "watch");
      return { count: t.length, traded: t.filter((s) => s.traded).length,
        avg: t.length ? Math.round(t.reduce((a, s) => a + s.confidence, 0) / t.length) : null };
    },

    async toggle(s) {
      if (this.expanded === s.id) { this.expanded = null; return; }
      this.expanded = s.id;
      if (!this.detail[s.id]) {
        try { this.detail[s.id] = await NT.api.get(`/signals/${s.id}`); } catch (e) { /* ignore */ }
      }
    },

    async approve(s) {
      if (!confirm(`Place a ${s.direction === "bullish" ? "BUY" : "SELL/SHORT"} for ${s.ticker} now? Risk checks still apply.`)) return;
      this.busy[s.id] = true;
      try {
        const r = await NT.api.post(`/signals/${s.id}/approve`);
        Alpine.store("nt").toast(r.traded ? "success" : "warn", r.traded ? `Order placed for ${s.ticker}` : `Not traded: ${s.ticker}`, r.reason);
        await Promise.all([this.load(true), this.loadReviews()]);
      } catch (e) { Alpine.store("nt").error(e, "Approve failed"); }
      finally { this.busy[s.id] = false; }
    },
    async dismiss(s) {
      try {
        await NT.api.post(`/signals/${s.id}/dismiss`);
        for (const x of [...this.reviews, ...this.signals]) if (x.id === s.id) x.review_status = "dismissed";
      }
      catch (e) { Alpine.store("nt").error(e); }
    },

    async runTest() {
      this.testing = true; this.testResult = null;
      try {
        const r = await NT.api.post("/ai/test", { text: this.testText });
        // the local engine explains every company it looked at, including the ones it called neutral
        r.neutral = [];
        if (r.engine === "local" && r.raw) {
          try {
            const shown = new Set((r.signals || []).map(s => s.ticker));
            r.neutral = (JSON.parse(r.raw).details || []).filter(d => !shown.has(d.ticker))
              .map(d => ({ ticker: d.ticker, event: d.event, why: d.neutral_because || "no clear good or bad news" }));
          } catch (e) { /* not JSON */ }
        }
        this.testResult = r;
        this.loadStats();
      }
      catch (e) { Alpine.store("nt").error(e, "AI test failed"); }
      finally { this.testing = false; }
    },

    actionClass(s) {
      if (s.action === "watch") return s.engine === "chart" ? "chart" : "pro";
      return { bought: "good", sold: "good", shorted: "good", review: "warn", blocked: "", ignored: "", error: "bad", merged: "" }[s.action] || "";
    },
    actionLabel(s) {
      if (s.traded) return { bought: "BOUGHT", sold: "SOLD", shorted: "SHORTED" }[s.action] || "TRADED";
      if (s.action === "review") return s.review_status === "dismissed" ? "dismissed" : "review";
      if (s.action === "watch") return s.engine === "chart" ? "Chart - watching" : "Pro AI - watching";
      return s.action;
    },
    // what the chart said about a news signal before it was traded (see newstrader/chart/service.py)
    chartChip(s) {
      if (!s || !s.chart_verdict) return null;
      const adj = s.chart_adjust ? ` ${s.chart_adjust > 0 ? "+" : ""}${s.chart_adjust}` : "";
      if (s.chart_verdict === "agrees") return { label: "Chart agrees" + adj, cls: "good" };
      if (s.chart_verdict === "against") return { label: "Chart disagrees" + adj, cls: "bad" };
      if (s.chart_verdict === "neutral") return { label: "Chart mixed", cls: "" };
      if (s.chart_verdict === "stretched") {
        // "Chart says the move may already be done: RSI 84 and 2.3 ATR above VWAP. Buying now would be chasing it."
        const m = /already be done: (.+?)\. (Buying|Selling) now/.exec(s.chart_reason || "");
        return { label: "Stretched" + (m ? ": " + m[1].split(" and ")[0] : ""), cls: "warn" };
      }
      return { label: "No chart", cls: "" };
    },
    showChart(s) {
      Alpine.store("nt").setTab("market");
      window.dispatchEvent(new CustomEvent("nt:open-chart", { detail: s.ticker }));
    },
    engineName(e) { return { local: "", claude: "Claude ", pro: "Pro AI " }[e] ?? ""; },
    // short labels for the reading flags the local engine attaches to a signal
    flagList(s) {
      let flags = s.flags || [];
      if (typeof flags === "string") { try { flags = JSON.parse(flags); } catch (e) { flags = []; } }
      const names = { unconfirmed: "not confirmed", denial: "denied", moved: "stock already moving",
                      country: "country fund", person_only: "person only" };
      return flags.filter(f => names[f]).map(f => names[f]);
    },
    dirClass(d) { return d === "bullish" ? "good" : d === "bearish" ? "bad" : ""; },
    confClass(c) { return c >= 80 ? "hi" : c >= 60 ? "mid" : "lo"; },
  }));
});
