// Signals tab: every AI signal, the manual-review queue, and a "test the AI" box.
document.addEventListener("alpine:init", () => {
  Alpine.data("signalsTab", () => ({
    fmt: NT.fmt,
    signals: [],
    filters: { ticker: "", direction: "", action: "", min_confidence: 0 },
    expanded: null,
    detail: {},
    stats: null,
    busy: {},
    testText: "",
    testing: false,
    testResult: null,

    async init() {
      window.addEventListener("nt:signal", (e) => this.upsert(e.detail));
      window.addEventListener("nt:signal_updated", () => { if (Alpine.store("nt").tab === "signals") this.load(true); });
      window.addEventListener("nt:tab", (e) => { if (e.detail === "signals") { this.load(); this.loadStats(); } });
      await this.load();
      this.loadStats();
    },

    async load(quiet = false) {
      const f = this.filters;
      const q = `limit=500&ticker=${encodeURIComponent(f.ticker)}&direction=${f.direction}&action=${f.action}&min_confidence=${f.min_confidence || 0}`;
      try { this.signals = (await NT.api.get("/signals?" + q)).signals; }
      catch (e) { if (!quiet) Alpine.store("nt").error(e, "Couldn't load signals"); }
    },
    async loadStats() { try { this.stats = await NT.api.get("/ai/stats"); } catch (e) { /* ignore */ } },

    upsert(s) {
      if (s.merged_into) return;
      if (typeof s.sources_seen === "string") { try { s.sources_seen = JSON.parse(s.sources_seen); } catch (e) { s.sources_seen = []; } }
      const i = this.signals.findIndex((x) => x.id === s.id);
      if (i >= 0) this.signals.splice(i, 1, s); else this.signals.unshift(s);
      this.loadStats();
    },

    get reviewQueue() { return this.signals.filter((s) => s.action === "review" && (!s.review_status || s.review_status === "pending")); },
    get today() {
      const d = new Date().toDateString();
      const t = this.signals.filter((s) => new Date(s.created_at).toDateString() === d);
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
        await this.load(true);
      } catch (e) { Alpine.store("nt").error(e, "Approve failed"); }
      finally { this.busy[s.id] = false; }
    },
    async dismiss(s) {
      try { await NT.api.post(`/signals/${s.id}/dismiss`); s.review_status = "dismissed"; }
      catch (e) { Alpine.store("nt").error(e); }
    },

    async runTest() {
      this.testing = true; this.testResult = null;
      try { this.testResult = await NT.api.post("/ai/test", { text: this.testText }); this.loadStats(); }
      catch (e) { Alpine.store("nt").error(e, "AI test failed"); }
      finally { this.testing = false; }
    },

    actionClass(a) {
      return { bought: "good", sold: "good", shorted: "good", review: "warn", blocked: "", ignored: "", error: "bad", merged: "" }[a] || "";
    },
    actionLabel(s) {
      if (s.traded) return { bought: "BOUGHT", sold: "SOLD", shorted: "SHORTED" }[s.action] || "TRADED";
      if (s.action === "review") return s.review_status === "dismissed" ? "dismissed" : "review";
      return s.action;
    },
    dirClass(d) { return d === "bullish" ? "good" : d === "bearish" ? "bad" : ""; },
    confClass(c) { return c >= 80 ? "hi" : c >= 60 ? "mid" : "lo"; },
  }));
});
