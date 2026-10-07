// Logs tab: system status, diagnostics, log viewer, rejected AI responses.
document.addEventListener("alpine:init", () => {
  Alpine.data("logsTab", () => ({
    fmt: NT.fmt,
    view: "logs",
    logs: [],
    level: "INFO",
    q: "",
    loading: false,
    paused: false,
    diag: null,
    diagRunning: false,
    rejections: [],
    expanded: null,

    async init() {
      await this.load();
      window.addEventListener("nt:log", (e) => {
        if (this.paused) return;
        const row = e.detail;
        const order = { DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50 };
        if ((order[row.level] || 20) < (order[this.level] || 20)) return;
        if (this.q && !(row.message + row.logger).toLowerCase().includes(this.q.toLowerCase())) return;
        this.logs.unshift(row);
        if (this.logs.length > 1000) this.logs.length = 1000;
      });
      window.addEventListener("nt:tab", (e) => { if (e.detail === "logs" && this.view === "rejections") this.loadRejections(); });
    },

    showOffSources: false,
    // turned-off news sources would fill the grid (there are 100+ presets) - they're hidden unless asked for
    get components() {
      const all = Alpine.store("nt").status.components || [];
      return this.showOffSources ? all : all.filter((c) => !(c.component.startsWith("source:") && c.level === "off"));
    },
    get hiddenSources() {
      return (Alpine.store("nt").status.components || []).filter((c) => c.component.startsWith("source:") && c.level === "off").length;
    },
    label(c) { return c.name || c.component; },

    async load() {
      this.loading = true;
      try {
        const r = await NT.api.get(`/logs?level=${this.level}&limit=400&q=${encodeURIComponent(this.q)}`);
        this.logs = r.logs;
      } catch (e) { Alpine.store("nt").error(e, "Couldn't load logs"); }
      finally { this.loading = false; }
    },

    async loadRejections() {
      try { this.rejections = (await NT.api.get("/rejections?limit=200")).rejections; }
      catch (e) { Alpine.store("nt").error(e); }
    },

    async runDiagnostics() {
      this.diagRunning = true;
      try { this.diag = await NT.api.post("/diagnostics"); }
      catch (e) { Alpine.store("nt").error(e, "Diagnostics failed"); }
      finally { this.diagRunning = false; }
    },

    async openFolder(which) {
      try { const r = await NT.api.post("/system/open-folder", { which }); Alpine.store("nt").toast("info", "Opened folder", r.path); }
      catch (e) { Alpine.store("nt").error(e); }
    },

    levelClass(l) { return { ERROR: "bad", CRITICAL: "bad", WARNING: "warn", INFO: "", DEBUG: "muted" }[l] || ""; },
  }));
});
