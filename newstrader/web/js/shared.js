// Backtest -> Local ML model -> Shared training: everyone's labels from GitHub, "Share my labels" and "Train with
// everyone's labels" (the student model). Nothing here can place a trade.
document.addEventListener("alpine:init", () => {
  Alpine.data("sharedTraining", () => ({
    fmt: NT.fmt,
    st: null,
    busy: false,
    progress: 0,
    message: "",
    shared: null,  // the last "Share my labels" result: file, where, steps

    init() {
      window.addEventListener("nt:tab", (e) => { if (e.detail === "backtest") this.load(); });
      window.addEventListener("nt:shared_labels", () => this.load());
      window.addEventListener("nt:student_training", (e) => {
        this.progress = e.detail.progress || 0;
        this.message = e.detail.message || "";
        if (e.detail.status && e.detail.status !== "running") this.load();
      });
      if (Alpine.store("nt").tab === "backtest") this.load();
    },

    async load() {
      try {
        this.st = await NT.api.get("/ml/shared");
        const p = this.st.progress;
        if (p && !this.message) { this.message = p.message || ""; this.progress = p.progress || 0; }
      } catch (e) { /* engine not running yet */ }
    },

    get report() { return this.st && this.st.student ? this.st.student.report : null; },
    get studentText() {
      const s = this.st && this.st.student;
      if (!s) return "not trained yet";
      const n = s.n_labels ? ` (learned from ${Number(s.n_labels).toLocaleString()} labels)` : "";
      if (!s.active) return "not used: " + s.why_inactive;
      return (s.bundled ? "in use - came with the app" : "in use") + n;
    },
    get studentPill() {
      const s = this.st && this.st.student;
      if (!s) return ["", ""];
      return s.active ? ["good", "student in use"] : ["", "student not used"];
    },
    get sharedLine() {
      const sh = this.st && this.st.shared;
      if (!sh) return "";
      const files = `${sh.files} file${sh.files === 1 ? "" : "s"} from ${sh.contributors} PC${sh.contributors === 1 ? "" : "s"}`;
      const when = sh.checked_at ? ` · checked ${this.fmt.dateTime(sh.checked_at)}` : " · not checked yet";
      return files + when + (this.st.pull_daily ? "" : " · daily check off");
    },
    get lastLine() {
      const r = this.st && this.st.last;
      if (!r) return "";
      if (r.error) return `Last training didn't finish: ${r.error}`;
      const u = r.used || {};
      const n = (v) => Number(v || 0).toLocaleString();
      let text = `Last training (${this.fmt.dateTime(r.at)}): ${n(u.own)} of your labels and ${n(u.shared)} shared ones ` +
        `from ${n(u.contributors)} PC${u.contributors === 1 ? "" : "s"} in all`;
      const skipped = (u.not_downloaded || 0) + (u.headline_differs || 0);
      if (skipped) text += ` (${n(u.not_downloaded)} were for headlines this PC hasn't downloaded, ${n(u.headline_differs)} had a different headline)`;
      text += `. It agreed with Pro AI on ${r.accuracy}% of the newest ${n(r.n_valid)} labels it never saw`;
      if (r.current_accuracy !== null && r.current_accuracy !== undefined) {
        text += r.compare_basis === "same_labels"
          ? `. On the ${n(r.compared_on)} of those the student in use never learned from: new ${r.new_accuracy}%, in use ${r.current_accuracy}%`
          : `. The student in use got ${r.current_accuracy}% on its own test when it was trained`;
      }
      return text + (r.switched ? ". The new one is in use now." : " - the new one did worse, so the one in use stays.");
    },

    async pull(quiet = false) {
      this.busy = true;
      try {
        const r = await NT.api.post("/ml/shared/pull");
        if (!quiet) {
          const added = r.added.length;
          Alpine.store("nt").toast(r.error ? "warn" : "success", "Everyone's labels",
            r.error || (added ? `${added} new file${added === 1 ? "" : "s"} (${r.labels.toLocaleString()} labels in all)` : "Nothing new."));
        }
      } catch (e) { if (!quiet) Alpine.store("nt").error(e, "Couldn't get everyone's labels"); }
      finally { this.busy = false; await this.load(); }
    },

    async train() {
      const msg = "Train the student model with your labels and everyone's?\n\nIt first gets any new shared labels, then " +
        "learns in the background (a few minutes) while trading continues. It's only used if it does at least as well " +
        "as the student in use.";
      if (!confirm(msg)) return;
      if (this.st && this.st.pull_daily) await this.pull(true);
      this.busy = true;
      try {
        await NT.api.post("/ml/student/train");
        this.message = "starting..."; this.progress = 0;
        await this.load();
      } catch (e) { Alpine.store("nt").error(e, "Training didn't start"); }
      finally { this.busy = false; }
    },

    async cancel() { try { await NT.api.post("/ml/student/cancel"); } catch (e) { Alpine.store("nt").error(e); } },

    async share() {
      const msg = "Share your labels?\n\nNewsTrader saves them to one file in your Downloads folder (no headline text, " +
        "no keys, no trades - just which way Pro AI read each story) and opens GitHub's upload page. Nothing is " +
        "uploaded until you do it on that page.";
      if (!confirm(msg)) return;
      this.busy = true;
      try { this.shared = await NT.api.post("/ml/shared/export"); }
      catch (e) { Alpine.store("nt").error(e, "Couldn't save your labels"); }
      finally { this.busy = false; }
    },

    async remove() {
      if (!confirm("Delete the student model? The engine goes back to FinBERT's reading for stories with no news event.")) return;
      try { await NT.api.del("/ml/student"); this.message = ""; await this.load(); }
      catch (e) { Alpine.store("nt").error(e); }
    },
  }));
});
