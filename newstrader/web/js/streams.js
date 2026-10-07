// Live tab: one transcript pane per live TV stream, with on/off switches and GPU status.
document.addEventListener("alpine:init", () => {
  Alpine.data("streamsPanel", () => ({
    fmt: NT.fmt,
    streams: [],
    lines: {},
    info: null,
    showOff: false,

    async init() {
      window.addEventListener("nt:transcript", (e) => {
        const l = e.detail;
        const list = this.lines[l.source_id] || (this.lines[l.source_id] = []);
        list.push(l);
        if (list.length > 200) list.splice(0, list.length - 200);
        this.$nextTick(() => this.scroll(l.source_id));
      });
      window.addEventListener("nt:status", (e) => {
        const c = e.detail;
        if (c.component === "transcriber" && this.info) this.info.transcriber = c;
        if (!c.component.startsWith("source:")) return;
        const s = this.streams.find((x) => x.id === c.component.slice(7));
        if (s) s.health = c;
      });
      window.addEventListener("nt:sources_changed", () => setTimeout(() => this.load(), 800));
      window.addEventListener("nt:settings_changed", () => this.load());
      await this.load();
    },

    async load() {
      try {
        this.info = await NT.api.get("/streams");
        this.streams = this.info.streams;
        for (const s of this.streams.filter((x) => x.enabled)) {
          if (!this.lines[s.id]) {
            const r = await NT.api.get(`/transcripts?source_id=${s.id}&limit=60`);
            this.lines[s.id] = r.lines;
            this.$nextTick(() => this.scroll(s.id));
          }
        }
      } catch (e) { /* engine starting */ }
    },

    get visibleStreams() { return this.showOff ? this.streams : this.streams.filter((s) => s.enabled); },
    get offCount() { return this.streams.filter((s) => !s.enabled).length; },

    scroll(id) {
      const el = document.getElementById("tx-" + id);
      if (!el) return;
      if (el.scrollHeight - el.scrollTop - el.clientHeight < 120) el.scrollTop = el.scrollHeight;
    },

    async toggle(s) {
      try { await NT.api.post(`/sources/${s.id}/toggle`, { enabled: !s.enabled }); s.enabled = !s.enabled; if (s.enabled) this.load(); }
      catch (e) { Alpine.store("nt").error(e); }
    },

    highlight(l) {
      // Escape, then highlight tickers/companies Claude may look at.
      const esc = (t) => t.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
      let html = esc(l.text || "");
      for (const k of (l.keywords || [])) {
        const re = new RegExp("(" + k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
        html = html.replace(re, '<mark class="kw">$1</mark>');
      }
      return html;
    },
  }));
});
