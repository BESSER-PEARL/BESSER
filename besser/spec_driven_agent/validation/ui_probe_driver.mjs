// Opens each screen of a built web frontend in a headless Chromium and records what loading it did.
//
// Run by ui_probe.py inside the run sandbox: `node ui_probe_driver.mjs <config.json>`. Node 22's
// standard library only (the global WebSocket speaks the DevTools protocol), so nothing is
// installed and no browser is downloaded. It boots the app's backend, serves the build output
// under its base path (index.html for page navigations, the backend for any other path the build
// does not hold), opens every screen in a fresh tab and prints the report so far, prefixed with
// MARKER, after each screen.

import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";

const MARKER = "BESSER_SCREEN_REPORT:";
const QUIET_MS = 500;
const RECHECK_MS = 250;
const MAX_TEXT = 400;
const MAX_REQUESTS = 40;
const MAX_MESSAGES = 10;
const MAX_LOG = 200000;
const PROXY_HEADER = "x-besser-screen-proxied";
const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".map": "application/json",
  ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".gif": "image/gif", ".webp": "image/webp", ".ico": "image/x-icon", ".txt": "text/plain",
  ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf", ".wasm": "application/wasm",
  ".webmanifest": "application/manifest+json",
};
// What the page shows: text or a visible control anywhere in the body (a portal renders
// outside the app's root), React Router's default error page and the links.
const RENDERED_JS = `(() => {
  const body = document.body;
  if (!body) return { rendered: false, links: [] };
  const text = (body.innerText || "").trim();
  const media = body.querySelector("img,svg,canvas,video,input,button,select,textarea,iframe,table");
  const lines = text.split("\\n").map((line) => line.trim()).filter(Boolean);
  const at = lines.indexOf("Unexpected Application Error!");
  return {
    rendered: text.length > 0 || media !== null,
    error_page: at >= 0 ? (lines[at + 1] || "") : null,
    links: Array.from(document.querySelectorAll("a[href]"), (a) => a.href).slice(0, 200),
  };
})()`;

const cfg = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const deadline = Date.now() + cfg.total_ms;
const children = [];
const report = { backend: null, screens: [], unvisited: [], timed_out: false, error: null };
let backendLog = "";
let browserProcess = null;
let finished = false;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const cut = (text) => String(text ?? "").slice(0, MAX_TEXT);

function emit() {
  // ASCII only: the parent decodes this with the platform's locale encoding.
  const line = JSON.stringify(report).replace(/[\u007f-\uffff]/g,
    (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"));
  process.stdout.write(MARKER + line + "\n");
}

function finish(code = 0) {
  if (!finished) {
    finished = true;
    emit();
  }
  for (const child of children) {
    try { child.kill("SIGKILL"); } catch { /* already gone */ }
  }
  process.exit(code);
}

setTimeout(() => {
  report.timed_out = true;
  finish();
}, Math.max(1000, cfg.total_ms + 2000)).unref();

function portFree(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.listen(port, "127.0.0.1", () => server.close(() => resolve(true)));
  });
}

function answers(port, readyPath) {
  return new Promise((resolve) => {
    const request = http.get({ host: "127.0.0.1", port, path: readyPath, timeout: 2000 }, (response) => {
      response.resume();
      resolve(true);
    });
    request.on("error", () => resolve(false));
    request.on("timeout", () => { request.destroy(); resolve(false); });
  });
}

async function startBackend(backend) {
  if (!(await portFree(backend.port))) return { status: "port_in_use", port: backend.port };
  const child = spawn(backend.argv[0], backend.argv.slice(1), {
    cwd: backend.cwd, env: { ...process.env, ...backend.env }, stdio: ["ignore", "pipe", "pipe"],
  });
  children.push(child);
  let exited = null;
  const keep = (data) => { backendLog = (backendLog + data).slice(-MAX_LOG); };
  child.stdout.on("data", keep);
  child.stderr.on("data", keep);
  child.on("exit", (code) => { exited = code ?? "signal"; });
  child.on("error", (error) => { exited = String(error); });
  const until = Math.min(Date.now() + backend.boot_ms, deadline);
  while (Date.now() < until) {
    if (exited !== null) return { status: "exited", port: backend.port, log: cut(backendLog.slice(-MAX_TEXT)) };
    if (await answers(backend.port, backend.ready_path)) return { status: "up", port: backend.port };
    await sleep(200);
  }
  return { status: "timeout", port: backend.port, log: cut(backendLog.slice(-MAX_TEXT)) };
}

// The bundle may name several local API addresses (a build-time one and a fallback): each
// free one forwards to the backend.
async function forward(ports, backendPort) {
  for (const port of ports) {
    if (port === backendPort || !(await portFree(port))) continue;
    const server = net.createServer((socket) => {
      const upstream = net.connect(backendPort, "127.0.0.1");
      socket.pipe(upstream).pipe(socket);
      socket.on("error", () => upstream.destroy());
      upstream.on("error", () => socket.destroy());
    });
    await new Promise((resolve) => {
      server.once("error", resolve);
      server.listen(port, "127.0.0.1", resolve);
    });
  }
}

function sendFile(response, file) {
  fs.readFile(file, (error, body) => {
    if (error) {
      response.writeHead(404);
      response.end();
      return;
    }
    response.writeHead(200, { "content-type": MIME[path.extname(file).toLowerCase()] || "application/octet-stream" });
    response.end(body);
  });
}

function proxy(request, response, backendPort) {
  const refuse = () => {
    if (!response.headersSent) response.writeHead(502, { [PROXY_HEADER]: "unreachable" });
    response.end();
  };
  if (!backendPort) {
    request.resume();
    refuse();
    return;
  }
  const upstream = http.request({
    host: "127.0.0.1", port: backendPort, method: request.method, path: request.url,
    headers: { ...request.headers, host: `127.0.0.1:${backendPort}` },
  }, (answer) => {
    response.writeHead(answer.statusCode, { ...answer.headers, [PROXY_HEADER]: "1" });
    answer.pipe(response);
  });
  upstream.on("error", refuse);
  request.pipe(upstream);
}

function staticServer(site, base, backendPort) {
  const root = path.resolve(site);
  const fileAt = (pathname) => {
    const file = path.resolve(root, "." + pathname);
    if (file !== root && !file.startsWith(root + path.sep)) return null;
    try { return fs.statSync(file).isFile() ? file : null; } catch { return null; }
  };
  return http.createServer((request, response) => {
    const url = new URL(request.url, "http://localhost");
    let pathname = url.pathname;
    try { pathname = decodeURIComponent(pathname); } catch { /* keep it encoded */ }
    // A build made for a base path (`vite build --base /app/`) asks for its files under it.
    const file = (pathname.startsWith(base) && fileAt(pathname.slice(base.length - 1))) || fileAt(pathname);
    if (file) {
      sendFile(response, file);
      return;
    }
    const navigation = request.headers["sec-fetch-mode"] === "navigate"
      || (request.headers.accept || "").includes("text/html");
    if (request.method === "GET" && navigation) {
      sendFile(response, path.join(root, "index.html"));
      return;
    }
    proxy(request, response, backendPort);
  });
}

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
}

async function launchBrowser() {
  const args = [
    "--headless", "--remote-debugging-port=0", `--user-data-dir=${cfg.profile}`,
    "--no-first-run", "--no-default-browser-check", "--disable-extensions",
    "--disable-background-networking", "--disable-component-update", "--disable-sync",
    "--disable-default-apps", "--mute-audio", "--no-proxy-server",
    // WebGL in software: the headless shell has no GPU to give it otherwise.
    "--enable-unsafe-swiftshader",
    // The deployed frontend and API origins are unknown here, so CORS is not judged.
    "--disable-web-security",
    // Same result with or without a network: nothing outside this machine is reached.
    "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1",
    "--window-size=1280,800", ...cfg.browser_args, "about:blank",
  ];
  const child = spawn(cfg.browser, args, { stdio: ["ignore", "ignore", "pipe"] });
  children.push(child);
  browserProcess = child;
  let stderr = "";
  let exited = null;
  child.stderr.on("data", (data) => { stderr = (stderr + data).slice(-8000); });
  child.on("exit", (code) => { exited = code ?? "signal"; });
  child.on("error", (error) => { exited = String(error); });
  const portFile = path.join(cfg.profile, "DevToolsActivePort");
  const until = Math.min(Date.now() + 30000, deadline);
  while (Date.now() < until) {
    const found = /DevTools listening on (ws:\/\/\S+)/.exec(stderr);
    if (found) return found[1];
    const lines = fs.existsSync(portFile) ? fs.readFileSync(portFile, "utf8").split(/\r?\n/) : [];
    if (lines.length >= 2 && lines[0] && lines[1]) return `ws://127.0.0.1:${lines[0]}${lines[1]}`;
    if (exited !== null) throw new Error(`the browser exited at startup: ${cut(stderr.trim().split("\n").pop())}`);
    await sleep(100);
  }
  throw new Error("the browser did not open its DevTools endpoint in time");
}

class Cdp {
  constructor(socket) {
    this.socket = socket;
    this.next = 0;
    this.pending = new Map();
    this.listeners = new Set();
    socket.onmessage = (event) => {
      const message = JSON.parse(typeof event.data === "string" ? event.data : Buffer.from(event.data).toString());
      if (message.id === undefined) {
        for (const listener of this.listeners) listener(message);
        return;
      }
      const waiter = this.pending.get(message.id);
      if (!waiter) return;
      this.pending.delete(message.id);
      if (message.error) waiter.reject(new Error(message.error.message));
      else waiter.resolve(message.result);
    };
    socket.onclose = () => {
      for (const waiter of this.pending.values()) waiter.reject(new Error("the browser connection closed"));
      this.pending.clear();
    };
  }

  send(method, params = {}, sessionId = undefined, timeoutMs = 10000) {
    const id = ++this.next;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${method} did not answer in time`));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: (value) => { clearTimeout(timer); resolve(value); },
        reject: (error) => { clearTimeout(timer); reject(error); },
      });
      this.socket.send(JSON.stringify(sessionId ? { id, method, params, sessionId } : { id, method, params }));
    });
  }
}

function connect(url) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    socket.onopen = () => resolve(socket);
    socket.onerror = () => reject(new Error("could not connect to the browser"));
  });
}

function header(headers, name) {
  for (const [key, value] of Object.entries(headers || {})) if (key.toLowerCase() === name) return value;
  return undefined;
}

function isLocal(url) {
  try {
    return ["localhost", "127.0.0.1"].includes(new URL(url).hostname);
  } catch {
    return false;
  }
}

async function visit(cdp, url, ownPorts) {
  const screen = {
    url, exceptions: [], console_errors: [], requests: [], rendered: null, error_page: null,
    idle: false, loaded: false, ms: 0, links: [], error: null, request_count: 0, backend_log: "",
  };
  const started = Date.now();
  const logFrom = backendLog.length;
  const sent = new Map();
  const inflight = new Set();
  let last = Date.now();
  let sessionId = null;
  const own = (requestUrl, headers) => header(headers, PROXY_HEADER) !== undefined
    || (isLocal(requestUrl) && ownPorts.has(new URL(requestUrl).port));
  const note = (list, item, limit) => { if (list.length < limit) list.push(item); };
  const listener = (message) => {
    if (message.sessionId !== sessionId) return;
    const p = message.params || {};
    switch (message.method) {
      case "Network.requestWillBeSent":
        if (!p.redirectResponse) screen.request_count += 1;
        sent.set(p.requestId, { method: p.request.method, url: p.request.url, type: p.type || "" });
        if (p.type !== "EventSource") inflight.add(p.requestId);
        last = Date.now();
        break;
      case "Network.responseReceived": {
        const status = p.response.status;
        const request = sent.get(p.requestId) || { method: "GET", type: p.type || "" };
        if (status >= 400) {
          note(screen.requests, {
            method: request.method, url: cut(p.response.url), type: request.type, status,
            own: own(p.response.url, p.response.headers),
            unreachable: header(p.response.headers, PROXY_HEADER) === "unreachable",
          }, MAX_REQUESTS);
        }
        break;
      }
      case "Network.loadingFinished":
        inflight.delete(p.requestId);
        last = Date.now();
        break;
      case "Network.loadingFailed": {
        inflight.delete(p.requestId);
        last = Date.now();
        const request = sent.get(p.requestId);
        if (request && !p.canceled) {
          note(screen.requests, {
            method: request.method, url: cut(request.url), type: request.type, error: cut(p.errorText),
            own: own(request.url), unreachable: isLocal(request.url) && own(request.url),
          }, MAX_REQUESTS);
        }
        break;
      }
      case "Runtime.exceptionThrown": {
        const details = p.exceptionDetails || {};
        note(screen.exceptions, {
          text: cut(details.exception?.description || details.exception?.value || details.text),
          promise: /in promise/i.test(details.text || ""),
          url: cut(details.url || details.stackTrace?.callFrames?.[0]?.url || ""),
        }, MAX_MESSAGES);
        break;
      }
      case "Runtime.consoleAPICalled":
        if (p.type === "error" || p.type === "assert") {
          note(screen.console_errors, cut((p.args || []).map((a) => a.value ?? a.description ?? "").join(" ")),
            MAX_MESSAGES);
        }
        break;
      case "Log.entryAdded":
        if (p.entry.level === "error" && p.entry.source !== "network") {
          note(screen.console_errors, cut(p.entry.text), MAX_MESSAGES);
        }
        break;
      case "Page.loadEventFired":
        screen.loaded = true;
        last = Date.now();
        break;
      default:
        break;
    }
  };
  cdp.listeners.add(listener);
  let targetId = null;
  const end = Math.min(started + cfg.screen_ms, deadline - 1000);
  const look = async () => {
    const evaluated = await cdp.send("Runtime.evaluate",
      { expression: RENDERED_JS, returnByValue: true }, sessionId);
    const value = evaluated.result?.value || {};
    screen.rendered = value.rendered === true;
    screen.error_page = value.error_page ?? null;
    screen.links = value.links || [];
  };
  try {
    ({ targetId } = await cdp.send("Target.createTarget", { url: "about:blank" }));
    ({ sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true }));
    for (const domain of ["Runtime", "Network", "Page", "Log"]) await cdp.send(`${domain}.enable`, {}, sessionId);
    const navigation = await cdp.send("Page.navigate", { url }, sessionId);
    if (navigation.errorText) screen.error = cut(navigation.errorText);
    while (Date.now() < end) {
      if (screen.loaded && inflight.size === 0 && Date.now() - last >= QUIET_MS) {
        screen.idle = true;
        break;
      }
      await sleep(50);
    }
    await look();
    // A first render can come later than the network goes quiet (a timer, a lazy chunk):
    // look again while the page is still empty, up to the screen's limit.
    while (!screen.rendered && Date.now() + RECHECK_MS < end) {
      await sleep(RECHECK_MS);
      await look();
    }
  } catch (error) {
    screen.error = cut(error.message);
  } finally {
    cdp.listeners.delete(listener);
    if (targetId) await cdp.send("Target.closeTarget", { targetId }).catch(() => {});
    screen.ms = Date.now() - started;
    screen.backend_log = backendLog.slice(logFrom).slice(-2000);
  }
  return screen;
}

function routeOf(link, origin) {
  let url;
  try { url = new URL(link); } catch { return null; }
  if (url.origin !== origin) return null;
  let route = url.pathname;
  if (cfg.hash) {
    if (!url.hash.startsWith("#/")) return null;
    route = url.hash.slice(1).split("?")[0];
  } else if (cfg.prefix) {
    if (route !== cfg.prefix && !route.startsWith(cfg.prefix + "/")) return null;
    route = route.slice(cfg.prefix.length);
  }
  if (/\.[A-Za-z0-9]{1,8}$/.test(route)) return null; // a file, not a screen
  route = route.replace(/\/+$/, "") || "/";
  return route;
}

async function main() {
  if (typeof WebSocket === "undefined") throw new Error("this Node.js has no WebSocket client (Node 22 or later is needed)");
  let backendPort = null;
  if (cfg.backend) {
    report.backend = await startBackend(cfg.backend);
    if (report.backend.status === "port_in_use") return;
    if (report.backend.status === "up") {
      backendPort = cfg.backend.port;
      await forward(cfg.own_ports, backendPort);
    }
  }
  const server = staticServer(cfg.site, cfg.base, backendPort);
  const port = await listen(server);
  const origin = `http://127.0.0.1:${port}`;
  const ownPorts = new Set(cfg.own_ports.map(String));
  const cdp = new Cdp(await connect(await launchBrowser()));
  const patterns = cfg.route_patterns.map((source) => new RegExp(source));
  const queue = [...cfg.routes];
  const seen = new Set(queue);
  while (queue.length && report.screens.length < cfg.max_screens) {
    if (Date.now() >= deadline - 2000) {
      report.timed_out = true;
      break;
    }
    const route = queue.shift();
    const target = cfg.hash ? `${cfg.prefix}/#${route}` : cfg.prefix + (route === "/" && cfg.prefix ? "" : route);
    const screen = await visit(cdp, origin + (target || "/"), ownPorts);
    screen.route = route;
    for (const link of screen.links) {
      const next = routeOf(link, origin);
      if (next && !seen.has(next) && (!patterns.length || patterns.some((p) => p.test(next)))) {
        seen.add(next);
        queue.push(next);
      }
    }
    delete screen.links;
    report.screens.push(screen);
    report.unvisited = [...queue];
    emit(); // what was observed so far survives the run being stopped
  }
  report.unvisited = queue;
  await cdp.send("Browser.close", {}, undefined, 3000).catch(() => {});
  // Let it release its profile before the parent deletes it.
  await new Promise((resolve) => {
    if (browserProcess.exitCode !== null || browserProcess.signalCode !== null) resolve();
    browserProcess.once("exit", resolve);
    setTimeout(resolve, 5000);
  });
  server.close();
}

main().then(() => finish(0), (error) => {
  report.error = cut(error?.message || error);
  finish(0);
});
