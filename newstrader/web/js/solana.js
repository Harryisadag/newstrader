// Solana (paper) tab: the memecoin desk's checklist feed, the paper wallet, positions, trades and the reject audit.
// PAPER ONLY - nothing on this tab can trade. It only reads the desk, pauses it, or starts a new paper wallet.
document.addEventListener("alpine:init", () => {
  const STATUS = {
    queued: ["waiting", ""], checking: ["checking...", "warn"], rejected: ["rejected", "bad"],
    bought: ["paper buy", "good"], passed: ["passed, not bought", "info"], skipped: ["skipped", ""],
    expired: ["too old", ""],
  };
  const EXIT = { stop_loss: "Stop-loss", trailing_stop: "Trailing stop", time_stop: "Time stop" };

  Alpine.data("solanaTab", () => ({
    fmt: NT.fmt,
    data: null,
    loading: false,
    busy: false,
    detail: null,
    detailOpen: false,
    resetOpen: false,
    resetPhrase: "",
    feedFilter: "all",

    init() {
      const active = () => Alpine.store("nt").tab === "solana";
      window.addEventListener("nt:tab", (e) => { if (e.detail === "solana") this.load(); });
      window.addEventListener("nt:solana", () => { if (active()) this.load(true); });
      setInterval(() => { if (active() && !document.hidden) this.load(true); }, 5000);
      if (active()) this.load();
    },

    async load(quiet = false) {
      if (!quiet) this.loading = true;
      try { this.data = await NT.api.get("/solana"); }
      catch (e) { if (!quiet) Alpine.store("nt").error(e, "Couldn't load the Solana paper desk"); }
      finally { this.loading = false; }
    },

    get s() { return (this.data && this.data.summary) || {}; },
    get w() { return (this.data && this.data.wallet) || {}; },
    get feed() {
      const rows = (this.data && this.data.feed) || [];
      if (this.feedFilter === "all") return rows;
      if (this.feedFilter === "bought") return rows.filter((r) => r.status === "bought" || r.status === "passed");
      return rows.filter((r) => r.status === this.feedFilter);
    },
    get positions() { return (this.data && this.data.positions) || []; },
    get closed() { return (this.data && this.data.closed) || []; },
    get audit() { return (this.data && this.data.audit) || { by_gate: [], rejected_then_ran: [], passed_then_rugged: [], totals: {}, passed: {} }; },
    get stuck() { return this.positions.filter((p) => p.cant_sell); },

    get stateText() {
      return { off: "OFF", watching: "WATCHING", paused: "PAUSED", killed: "NO NEW BUYS (kill switch)",
        halted: "HALTED TODAY", problem: "PROBLEM" }[this.s.state] || "—";
    },
    get stateClass() {
      return { watching: "good", paused: "warn", killed: "bad", halted: "warn", problem: "bad" }[this.s.state] || "";
    },

    sol(v, digits = 4) {
      if (v === null || v === undefined || isNaN(v)) return "—";
      return Number(v).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits }) + " SOL";
    },
    signedSol(v, digits = 4) {
      if (v === null || v === undefined || isNaN(v)) return "—";
      return (Number(v) >= 0 ? "+" : "") + this.sol(v, digits);
    },
    pctText(v) { return v === null || v === undefined ? "—" : this.fmt.pct(v, 1); },
    statusLabel(st) { return (STATUS[st] || [st])[0]; },
    statusClass(st) { return (STATUS[st] || ["", ""])[1]; },
    lightClass(l) { return { pass: "pass", fail: "fail" }[l.state] || "skip"; },
    lightTip(l) {
      const st = { pass: "passed", fail: "FAILED", not_run: "not checked" }[l.state] || l.state;
      return `Gate ${l.gate} - ${l.name}: ${st}` + (l.reason ? `\n${l.reason}` : "");
    },
    exitLabel(p) {
      const kind = String(p.exit_reason || "").split(":")[0];
      if (p.status === "old_wallet") return "Old wallet (never sold)";
      return EXIT[kind] || kind || "—";
    },
    valueTip(p) {
      if (p.value_kind === "estimate") {
        return "An ESTIMATE: Jupiter found no way to sell it, so this is the price feed x your coins x 0.95. Nothing is sold until a real sell quote comes back.";
      }
      return "What a Jupiter quote to sell it all would give right now, after the 1% haircut and the fee.";
    },
    levelsText(p) {
      const lv = p.levels || {};
      const parts = [`stop at ${this.sol(lv.stop_value)}`];
      parts.push(lv.trailing ? `trailing: sells at ${this.sol(lv.trail_value)}` : `trailing stop from ${this.sol(lv.trail_start_value)}`);
      if (lv.time_stop_at) parts.push(`time stop ${this.fmt.dateTime(lv.time_stop_at)}`);
      return parts.join(" · ");
    },
    sourceLine(src) {
      if (!src.calls) return "not asked yet";
      if (src.error) return `${src.error} (${this.fmt.ago(src.error_at)})`;
      return `answered ${this.fmt.ago(src.ok_at)}` + (src.limited ? ` · ${src.limited} slow-downs` : "");
    },

    async show(row) {
      try {
        this.detail = await NT.api.get(`/solana/candidates/${row.id}`);
        this.detailOpen = true;
      } catch (e) { Alpine.store("nt").error(e, "Couldn't load that coin"); }
    },
    get attempts() {
      const out = {};
      for (const g of ((this.detail && this.detail.gates) || [])) (out[g.attempt] = out[g.attempt] || []).push(g);
      return Object.entries(out).map(([n, gates]) => ({ n: Number(n), gates }));
    },
    dataLine(g) {
      const d = g.data || {};
      const keep = Object.entries(d).filter(([k, v]) => v !== null && typeof v !== "object" && k !== "stale_only");
      return keep.map(([k, v]) => `${k.replace(/_/g, " ")}: ${typeof v === "number" ? Number(v.toPrecision(6)) : v}`).join(" · ");
    },

    async act(path, done) {
      this.busy = true;
      try {
        const r = await NT.api.post(path);
        if (done) Alpine.store("nt").toast("success", done(r));
        await this.load(true);
        window.dispatchEvent(new CustomEvent("nt:refresh-status"));
      } catch (e) { Alpine.store("nt").error(e); }
      finally { this.busy = false; }
    },
    pause() { return this.act("/solana/pause", () => "Solana paper desk paused - open paper positions are still watched"); },
    resume() { return this.act("/solana/resume", () => "Solana paper desk resumed"); },
    checkNow() {
      return this.act("/solana/check-now", (r) => `Feed checked: ${r.pools} new pools, ${r.looked_at} looked at again, ${r.triggered} picked` + (r.error ? ` (${r.error})` : ""));
    },
    openReset() { this.resetPhrase = ""; this.resetOpen = true; },
    async reset() {
      this.busy = true;
      try {
        await NT.api.post("/solana/reset", { phrase: this.resetPhrase.trim() });
        this.resetOpen = false;
        Alpine.store("nt").toast("success", "New paper wallet", "Old paper trades stay in the list, marked \"old wallet\".");
        await this.load(true);
      } catch (e) { Alpine.store("nt").error(e, "The paper wallet was not reset"); }
      finally { this.busy = false; }
    },
  }));
});
