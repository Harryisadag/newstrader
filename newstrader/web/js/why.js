// "Why did it trade?" card: opened from a signal (Signals tab) or a trade (Trade Log) with
// window.dispatchEvent(new CustomEvent("nt:why", { detail: { signal_id } or { order_id } })).
document.addEventListener("alpine:init", () => {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  let whyChart = null;

  const RESULT = {
    pass: { icon: "✓", cls: "good", word: "passed" },
    review: { icon: "!", cls: "warn", word: "sent to manual review" },
    stop: { icon: "✕", cls: "bad", word: "stopped it" },
    info: { icon: "•", cls: "", word: "" },
  };
  const OUTCOME = {
    bought: ["BOUGHT", "good"], sold: ["SOLD", "good"], shorted: ["SHORTED", "good"], review: ["MANUAL REVIEW", "warn"],
    blocked: ["BLOCKED", "bad"], ignored: ["NOT TRADED", ""], error: ["ERROR", "bad"], watch: ["WATCHING ONLY", ""],
    merged: ["REPEAT OF ANOTHER SIGNAL", ""], pending: ["DECIDING...", ""],
  };
  const CHART_WORDS = { agrees: "The chart agreed", against: "The chart disagreed", neutral: "The chart was mixed",
    stretched: "The chart said the move may already be over", unavailable: "No up-to-date chart", skipped: "Chart check" };

  Alpine.data("whyCard", () => ({
    fmt: NT.fmt,
    visible: false,
    loading: false,
    error: "",
    card: null,
    target: null,
    busy: false,

    async open(target) {
      this.target = target || {};
      this.visible = true;
      this.card = null;
      this.error = "";
      this.destroyChart();
      this.loading = true;
      try {
        const path = this.target.order_id ? `/trades/${this.target.order_id}/why` : `/signals/${this.target.signal_id}/why`;
        this.card = await NT.api.get(path);
        this.$nextTick(() => this.drawChart());
      } catch (e) { this.error = e.message; }
      finally { this.loading = false; }
    },
    close() { this.visible = false; this.destroyChart(); },
    destroyChart() { if (whyChart) { whyChart.destroy(); whyChart = null; } },

    get sig() { return (this.card && this.card.signal) || null; },
    get trail() { return (this.card && this.card.trail) || null; },
    get title() {
      const s = this.sig;
      if (s) return `${s.ticker} · ${s.direction} · confidence ${s.confidence}`;
      const o = this.card && this.card.order;
      return o ? `${o.symbol} · ${o.side} order` : "";
    },
    get outcome() {
      const s = this.sig;
      if (!s) return null;
      const [label, cls] = OUTCOME[s.action] || [String(s.action || "").toUpperCase(), ""];
      let text = label;
      if (s.action === "review" && s.review_status === "dismissed") text = "DISMISSED BY YOU";
      if (s.action === "review" && s.review_status === "approved") text = "APPROVED BY YOU";
      return { label: text, cls, reason: s.action_reason || "" };
    },
    get canApprove() {
      const s = this.sig;
      return !!s && !s.traded && ["bullish", "bearish"].includes(s.direction) && !["watch", "merged"].includes(s.action);
    },
    result(r) { return RESULT[r] || RESULT.info; },
    chartWords(v) { return CHART_WORDS[v] || "Chart check"; },
    styleName(k) { return { cautious: "Cautious", normal: "Normal", aggressive: "Aggressive" }[k] || "Custom"; },
    get settingsLine() {
      const t = this.trail;
      if (!t || !t.settings) return "";
      const x = t.settings;
      return `Style ${this.styleName(t.style)} · buy at ${x.buy_threshold}+ · manual review ${x.review_threshold}-${x.buy_threshold - 1} · ` +
        `stop-loss ${x.stop_loss_pct}% · take-profit ${x.take_profit_pct}% · don't chase ${x.max_chase_pct ? x.max_chase_pct + "%" : "off"} · ` +
        `chart check ${x.chart_confirm}`;
    },
    get accountLine() {
      const t = this.trail;
      if (!t || !t.account) return "";
      if (t.account.mode === "live") return "LIVE account (real money)";
      return t.account.broker === "simulator" ? "Built-in simulator (fake money)" : "Alpaca paper account";
    },
    confStep(c, i) {
      if (i === 0) return `${c.label}: ${c.value}`;
      return `${c.label}: ${c.change > 0 ? "+" : ""}${c.change} → ${c.value}`;
    },
    earlierLine(e) {
      const o = e.outcome || {};
      const [label] = OUTCOME[o.action] || [String(o.action || "")];
      return `${this.fmt.dateTime(e.decided_at)}: ${label}${e.manual ? " (approved by you)" : ""} - ${o.reason || ""}`;
    },
    get tradeLine() {
      const t = this.card && this.card.trade;
      if (!t) return "";
      const verb = { open_long: "Bought", open_short: "Shorted", close_long: "Sold", close_short: "Covered" }[t.intent] || "Order";
      const price = t.entry_price ? ` at ${this.fmt.price(t.entry_price)}` : (t.estimated_price ? ` (about ${this.fmt.price(t.estimated_price)} - not filled yet)` : "");
      return `${verb} ${this.fmt.qty(t.qty)} ${t.symbol}${price}`;
    },
    pct(v) { return v === null || v === undefined ? "—" : this.fmt.pct(v); },
    vsWord(v) {
      if (v === null || v === undefined) return "";
      if (Math.abs(v) < 0.05) return "same as the market";
      return v > 0 ? "better than the market" : "worse than the market";
    },

    async approve() {
      const s = this.sig;
      if (!s || !confirm(`Place a ${s.direction === "bullish" ? "BUY" : "SELL/SHORT"} for ${s.ticker} now? Risk checks still apply.`)) return;
      this.busy = true;
      try {
        const r = await NT.api.post(`/signals/${s.id}/approve`);
        Alpine.store("nt").toast(r.traded ? "success" : "warn", r.traded ? `Order placed for ${s.ticker}` : `Not traded: ${s.ticker}`, r.reason);
        await this.open({ signal_id: s.id });
      } catch (e) { Alpine.store("nt").error(e, "Approve failed"); }
      finally { this.busy = false; }
    },
    showChart() {
      const sym = (this.sig && this.sig.ticker) || (this.card && this.card.order && this.card.order.symbol);
      this.close();
      Alpine.store("nt").setTab("market");
      window.dispatchEvent(new CustomEvent("nt:open-chart", { detail: sym }));
    },

    // price from 30 minutes before the news to 2 hours after, with the news, entry, exit, stop and target marked
    drawChart() {
      const c = this.card && Alpine.raw(this.card.chart);
      const el = document.getElementById("why-chart");
      if (!el || !c || !c.t.length) { this.destroyChart(); return; }
      const z = this.fmt.zone();
      const times = c.t.map((t) => new Date(t).getTime());
      const labels = c.t.map((t) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", timeZone: z.timeZone }));
      const nearest = (iso) => {
        const x = new Date(iso).getTime();
        let best = 0;
        for (let i = 0; i < times.length; i++) if (Math.abs(times[i] - x) < Math.abs(times[best] - x)) best = i;
        return Math.abs(times[best] - x) <= 5 * 60 * 1000 ? best : -1;
      };
      const flat = (v) => labels.map(() => v);
      const colors = { news: css("--accent"), entry: css("--good"), exit: css("--warn") };
      const datasets = [{ label: "Price", data: c.close.slice(), borderColor: css("--text"), borderWidth: 1.5, pointRadius: 0, tension: 0.1, spanGaps: true }];
      for (const m of c.markers || []) {
        const i = nearest(m.t);
        if (i < 0 || m.price === null || m.price === undefined) continue;
        const data = labels.map(() => null);
        data[i] = m.price;
        datasets.push({ label: `${m.label} ${this.fmt.price(m.price)}`, data, borderColor: colors[m.kind] || css("--muted"),
          backgroundColor: colors[m.kind] || css("--muted"), pointRadius: 7, pointHoverRadius: 8, pointStyle: m.kind === "news" ? "triangle" : "circle", showLine: false });
      }
      if (c.stop) datasets.push({ label: "Stop-loss " + this.fmt.price(c.stop), data: flat(c.stop), borderColor: css("--bad"), borderWidth: 1, borderDash: [5, 4], pointRadius: 0 });
      if (c.target) datasets.push({ label: "Take-profit " + this.fmt.price(c.target), data: flat(c.target), borderColor: css("--good"), borderWidth: 1, borderDash: [5, 4], pointRadius: 0 });
      this.destroyChart();
      whyChart = new Chart(el, {
        type: "line",
        data: { labels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          interaction: { mode: "nearest", intersect: false },
          plugins: {
            legend: { labels: { color: css("--muted"), boxWidth: 12 } },
            tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label.split(" $")[0]}: ${NT.fmt.price(ctx.parsed.y)}` } },
          },
          scales: {
            x: { ticks: { color: css("--muted"), maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.04)" } },
            y: { ticks: { color: css("--muted"), callback: (v) => NT.fmt.price(v) }, grid: { color: "rgba(255,255,255,0.06)" } },
          },
        },
      });
    },
  }));
});
