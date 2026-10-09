// NewsTrader front-end helpers: API calls (with the launch token), live websocket, formatting.
(function () {
  const params = new URLSearchParams(location.search);
  const urlToken = params.get("token");
  if (urlToken) {
    sessionStorage.setItem("nt_token", urlToken);
    // keep the token out of the visible URL
    history.replaceState(null, "", location.pathname);
  }
  const token = sessionStorage.getItem("nt_token") || "";

  class ApiError extends Error {
    constructor(message, status, details) {
      super(message);
      this.status = status;
      this.details = details || [];
    }
  }

  async function request(method, path, body) {
    const opts = { method, headers: { "X-NT-Token": token } };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    let res;
    try {
      res = await fetch("/api" + path, opts);
    } catch (e) {
      throw new ApiError("Can't reach the NewsTrader engine. Is it still running?", 0);
    }
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { data = text; }
    if (!res.ok) {
      const msg = (data && (data.detail || data.error)) || `Request failed (${res.status})`;
      throw new ApiError(typeof msg === "string" ? msg : JSON.stringify(msg), res.status, data && data.details);
    }
    return data;
  }

  const api = {
    token,
    get: (p) => request("GET", p),
    post: (p, b) => request("POST", p, b === undefined ? {} : b),
    put: (p, b) => request("PUT", p, b),
    del: (p) => request("DELETE", p),
    url: (p) => "/api" + p + (p.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token),
  };

  // ---- live events over websocket, re-dispatched as window events "nt:<type>" ----
  let ws = null;
  let retry = 1000;
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws?token=${encodeURIComponent(token)}`);
    ws.onopen = () => {
      retry = 1000;
      window.dispatchEvent(new CustomEvent("nt:connection", { detail: { connected: true } }));
    };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      window.dispatchEvent(new CustomEvent("nt:" + msg.type, { detail: msg.data }));
    };
    ws.onclose = () => {
      window.dispatchEvent(new CustomEvent("nt:connection", { detail: { connected: false } }));
      setTimeout(connect, retry);
      retry = Math.min(retry * 2, 15000);
    };
  }

  // ---- formatting ----
  const fmt = {
    money(v, digits = 2) {
      if (v === null || v === undefined || isNaN(v)) return "—";
      const n = Number(v);
      return (n < 0 ? "-$" : "$") + Math.abs(n).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
    },
    signedMoney(v) {
      if (v === null || v === undefined || isNaN(v)) return "—";
      return (Number(v) >= 0 ? "+" : "") + fmt.money(v);
    },
    pct(v, digits = 2) {
      if (v === null || v === undefined || isNaN(v)) return "—";
      const n = Number(v);
      return (n > 0 ? "+" : "") + n.toFixed(digits) + "%";
    },
    num(v, digits = 0) {
      if (v === null || v === undefined || isNaN(v)) return "—";
      return Number(v).toLocaleString(undefined, { maximumFractionDigits: digits });
    },
    // 1234567 -> "1.2M" (share volumes, trade counts)
    compact(v) {
      if (v === null || v === undefined || isNaN(v)) return "—";
      const n = Number(v);
      const a = Math.abs(n);
      for (const [size, unit] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]]) {
        if (a >= size) return (n / size).toFixed(1).replace(/\.0$/, "") + unit;
      }
      return String(Math.round(n));
    },
    // Settings -> Display -> Show times in: this computer's zone, New York (the US market) or UTC
    zone() {
      const st = window.Alpine && Alpine.store("nt");
      const z = st && st.status && st.status.time_zone;
      if (z === "market") return { timeZone: "America/New_York", label: " ET" };
      if (z === "utc") return { timeZone: "UTC", label: " UTC" };
      return { timeZone: undefined, label: "" };
    },
    time(iso) {
      if (!iso) return "—";
      const z = fmt.zone();
      return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit",
        timeZone: z.timeZone }) + z.label;
    },
    dateTime(iso) {
      if (!iso) return "—";
      const z = fmt.zone();
      return new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
        timeZone: z.timeZone }) + z.label;
    },
    ago(iso) {
      if (!iso) return "never";
      const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
      if (s < 60) return Math.floor(s) + "s ago";
      if (s < 3600) return Math.floor(s / 60) + "m ago";
      if (s < 86400) return Math.floor(s / 3600) + "h ago";
      return Math.floor(s / 86400) + "d ago";
    },
    // Only http(s) links may be clicked - news feeds are untrusted.
    url(u) {
      return typeof u === "string" && /^https?:\/\//i.test(u.trim()) ? u.trim() : null;
    },
    plClass(v) {
      if (v === null || v === undefined || isNaN(v) || Number(v) === 0) return "";
      return Number(v) > 0 ? "pos" : "neg";
    },
    // 2.1 -> "2.1 s", 38 -> "38 s", 300 -> "5 min"
    secs(v) {
      if (v === null || v === undefined || isNaN(v)) return "—";
      const n = Math.max(0, Number(v));
      if (n < 10) return n.toFixed(1) + " s";
      if (n < 120) return Math.round(n) + " s";
      if (n < 7200) return Math.round(n / 60) + " min";
      if (n < 172800) return (n / 3600).toFixed(1).replace(/\.0$/, "") + " h";
      return Math.round(n / 86400) + " days";
    },
    // The speed timer on a signal or trade, in one line (see newstrader/performance/speed.py)
    speed(sp) {
      if (!sp) return "";
      const has = (v) => v !== null && v !== undefined;
      const old = has(sp.feed_delay_s) ? ` (the news was ${fmt.secs(sp.feed_delay_s)} old when it arrived)` : "";
      if (has(sp.news_to_order_s)) {
        return `${fmt.secs(sp.news_to_order_s)} from seeing the news to the order` +
          (sp.manual ? " - it waited for your approval" : "") + old;
      }
      if (has(sp.thinking_s)) return `Decided ${fmt.secs(sp.thinking_s)} after seeing the news${old}`;
      return has(sp.feed_delay_s) ? `The news was ${fmt.secs(sp.feed_delay_s)} old when it arrived` : "";
    },
    speedTip(sp) {
      if (!sp) return "";
      const step = (label, v) => (v === null || v === undefined ? null : `${label}: ${fmt.secs(v)}`);
      return [step("News came out → NewsTrader saw it (how late the website or TV clip was)", sp.feed_delay_s),
        step("NewsTrader saw it → the AI decided", sp.thinking_s),
        step("The AI decided → the order was placed", sp.order_s)].filter(Boolean).join("\n");
    },
  };

  window.NT = { api, fmt, ApiError, connect };
})();
