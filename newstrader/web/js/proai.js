// Settings -> Pro AI: status, "Set up Pro AI" (one-time download), start / stop, "Test my PC", delete downloads.
// Lives inside the settings tab, so get()/set() and proFits are the settings form's.
document.addEventListener("alpine:init", () => {
  const STATE_LABELS = { off: "Off", not_set_up: "Not set up", downloading: "Downloading", starting: "Starting",
    ready: "Running", error: "Problem" };
  const LEVELS = { ready: "ok", starting: "starting", downloading: "starting", error: "error", not_set_up: "warn", off: "off" };

  Alpine.data("proAI", () => ({
    fmt: NT.fmt,
    st: null,
    plan: null,
    planLoading: false,
    showSetup: false,
    busy: false,
    _onEvent: null,

    async init() {
      this._onEvent = (e) => {
        if (!this.st) return;
        const before = this.st.state;
        const testing = !!this.st.selftest_progress;
        Object.assign(this.st, e.detail);
        if (e.detail.state !== before || (testing && !e.detail.selftest_progress)) this.load();
      };
      window.addEventListener("nt:pro_ai", this._onEvent);
      // picking another model or server version while "Set up" is open shows its sizes straight away
      this.$watch("form && [form.ai.pro_ai.model, form.ai.pro_ai.build].join()", () => { if (this.showSetup) this.loadPlan(); });
      await this.load();
      if (this.st && this.st.disk_bytes) this.loadPlan(false);  // model list: which ones fit this PC
    },
    destroy() { window.removeEventListener("nt:pro_ai", this._onEvent); },

    async load() {
      try { this.st = await NT.api.get("/pro-ai"); } catch (e) { /* engine still starting */ }
    },
    async loadPlan(showErrors = true) {
      this.planLoading = true;
      const q = `?build=${encodeURIComponent(this.get("ai.pro_ai.build") || "")}&model=${encodeURIComponent(this.get("ai.pro_ai.model") || "")}`;
      try {
        this.plan = await NT.api.get("/pro-ai/plan" + q);
        this.proFits = Object.fromEntries(this.plan.models.map((m) => [m.key, m.fits]));
      } catch (e) { if (showErrors) Alpine.store("nt").error(e, "Couldn't look at this PC"); }
      finally { this.planLoading = false; }
    },
    async openSetup() { this.showSetup = true; await this.loadPlan(); },

    gb(bytes) { return bytes === null || bytes === undefined ? "—" : (bytes / 1e9).toFixed(1) + " GB"; },
    get stateLabel() { return this.st ? STATE_LABELS[this.st.state] || this.st.state : "—"; },
    get level() { return this.st ? LEVELS[this.st.state] || "warn" : "off"; },
    get downloaded() { return !!(this.st && this.st.downloaded && this.st.downloaded.server && this.st.downloaded.model); },
    get downloading() { return !!(this.st && this.st.state === "downloading"); },
    get progressPct() {
      const p = this.st && this.st.progress;
      return p && p.total ? Math.min(100, Math.round((100 * p.done) / p.total)) : 0;
    },
    get setupModel() { return this.plan ? this.plan.models.find((m) => m.key === this.plan.model) || null : null; },
    get setupBuild() { return this.plan ? this.plan.builds.find((b) => b.key === this.plan.build) || null : null; },
    get agreement() {
      const a = (this.st && this.st.agreement) || {};
      const both = (a.agree || 0) + (a.disagree || 0);
      return both ? `Agreed with the main engine on ${a.agree || 0} of ${both} stories both read` +
        (a.no_main ? ` (and read ${a.no_main} the main engine skipped)` : "") : "";
    },
    get test() { return this.st && this.st.selftest; },
    get testVerdict() {
      const t = this.test;
      if (!t || t.error || !t.total) return "";
      let speed = "Slow on this PC - try a smaller model.";
      if (t.fully_on_gpu === false) speed = "Part of the model runs on the processor - pick a smaller model for full speed.";
      else if (t.avg_seconds <= 3) speed = "Fast enough for live news.";
      else if (t.avg_seconds <= 8) speed = "OK for watching; a smaller model would be faster.";
      const share = t.right / t.total;
      const quality = share >= 0.85 ? "" : share >= 0.7 ? " It read a few headlines wrong."
        : " It read many headlines wrong - a bigger model reads news better, if one fits.";
      return speed + quality;
    },
    yesNo(v) { return v === true ? "yes" : v === false ? "no" : "can't tell"; },

    async confirmSetup() {
      const m = this.setupModel, b = this.setupBuild;
      if (!m || !b) return;
      this.busy = true;
      try {
        await NT.api.post("/pro-ai/setup", { build: b.key, model: m.key });
        this.showSetup = false;
        await this.load();
      } catch (e) { Alpine.store("nt").error(e, "Set-up didn't start"); }
      finally { this.busy = false; }
    },
    async cancelSetup() { try { await NT.api.post("/pro-ai/setup/cancel"); } catch (e) { Alpine.store("nt").error(e); } },
    async start() {
      try { await NT.api.post("/pro-ai/start"); await this.load(); } catch (e) { Alpine.store("nt").error(e, "Pro AI didn't start"); }
    },
    async stop() {
      try { await NT.api.post("/pro-ai/stop"); await this.load(); } catch (e) { Alpine.store("nt").error(e); }
    },
    async runTest() {
      try { await NT.api.post("/pro-ai/selftest"); await this.load(); } catch (e) { Alpine.store("nt").error(e, "Test didn't start"); }
    },
    async removeDownloads() {
      if (!confirm(`Delete Pro AI's downloads (${this.gb(this.st.disk_bytes)})? Pro AI is turned off; you can set it up again any time.`)) return;
      this.busy = true;
      try {
        const r = await NT.api.del("/pro-ai/downloads");
        Alpine.store("nt").toast("success", "Pro AI downloads deleted", `${this.gb(r.freed_bytes)} freed`);
        this.plan = null; this.proFits = {};
        await this.load();
      } catch (e) { Alpine.store("nt").error(e, "Couldn't delete the downloads"); }
      finally { this.busy = false; }
    },
  }));
});
