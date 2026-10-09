// Live tab: headline feed from the text sources (live TV transcript panes are in streams.js).
document.addEventListener("alpine:init", () => {
  Alpine.data("liveTab", () => ({
    fmt: NT.fmt,
    news: [],
    sources: [],
    onlyRelevant: true,
    sourceFilter: "",
    stats: null,
    paused: false,

    async init() {
      window.addEventListener("nt:news", (e) => {
        if (this.paused) return;
        const n = e.detail;
        if (n.kind === "transcript") return;
        this.news.unshift(n);
        if (this.news.length > 400) this.news.length = 400;
      });
      window.addEventListener("nt:analysis", (e) => {
        const n = this.news.find((x) => x.id === e.detail.item_id);
        if (n) n.status = e.detail.status === "ok" ? "analysed" : e.detail.status;
      });
      window.addEventListener("nt:signal", (e) => {
        const s = e.detail;
        if (s.action === "watch") return;  // Pro AI only watching: not a signal for this story
        const n = this.news.find((x) => x.title && s.headline && x.title.startsWith(s.headline.slice(0, 60)));
        if (n) { n.signals = [...(n.signals || []), s]; n.status = "signal"; }
      });
      window.addEventListener("nt:sources_changed", () => this.loadSources());
      window.addEventListener("nt:status", (e) => {
        const c = e.detail;
        if (!c.component.startsWith("source:")) return;
        const s = this.sources.find((x) => x.id === c.component.slice(7));
        if (s) s.health = c;
      });
      window.addEventListener("nt:tab", (e) => { if (e.detail === "live") this.loadStats(); });
      await Promise.all([this.loadNews(), this.loadSources(), this.loadStats()]);
      setInterval(() => { if (Alpine.store("nt").tab === "live") this.loadStats(); }, 10000);
    },

    async loadNews() {
      try { this.news = (await NT.api.get("/news?limit=300&kind=text")).news; } catch (e) { /* engine starting */ }
    },
    async loadSources() {
      try {
        const r = await NT.api.get("/sources");
        this.sources = r.sources.filter((s) => s.type !== "stream");
      } catch (e) { /* ignore */ }
    },
    async loadStats() { try { this.stats = await NT.api.get("/ai/stats"); } catch (e) { /* ignore */ } },

    async toggle(src) {
      try { await NT.api.post(`/sources/${src.id}/toggle`, { enabled: !src.enabled }); src.enabled = !src.enabled; }
      catch (e) { Alpine.store("nt").error(e); }
    },

    get visible() {
      return this.news.filter((n) => (!this.onlyRelevant || !["filtered", "seen", "stale"].includes(n.status))
        && (!this.sourceFilter || n.source_id === this.sourceFilter));
    },

    statusLabel(s) {
      return { filtered: "not relevant", duplicate: "same story", queued: "analysing...", analysed: "no trade signal",
        signal: "signal", rejected: "AI rejected", refusal: "AI declined", error: "error", skipped_cap: "spend cap",
        stale: "too old", dropped: "dropped", received: "new", no_tickers: "waiting for ticker list" }[s] || s;
    },
    statusClass(s) {
      return { signal: "good", queued: "info", analysed: "", duplicate: "", filtered: "", rejected: "warn",
        refusal: "warn", error: "bad", skipped_cap: "warn" }[s] || "";
    },
  }));
});
