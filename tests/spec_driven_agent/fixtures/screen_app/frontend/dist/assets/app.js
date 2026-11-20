// Stand-in for a bundled single-page app: one behaviour per screen, chosen by the path.
// Served under the base its script tag names ("/" or e.g. "/app/").
const base = new URL(document.querySelector("script[src]").src).pathname.replace(/assets\/app\.js$/, "");
const root = document.getElementById("root");
const route = ("/" + location.pathname.slice(base.length)).replace(/\/+$/, "") || "/";

function show(text, parent = root) {
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  parent.appendChild(paragraph);
}

function load(path) {
  return fetch(path, { headers: { accept: "application/json" } })
    .then((response) => show(`${path}: ${response.status}`))
    .catch(() => show(`${path}: unreachable`));
}

if (route === "/") {
  show("Orders");
  const link = document.createElement("a");
  link.href = base + "orders/7";
  link.textContent = "Order 7";
  root.appendChild(link);
  load("/api/orders");
} else if (route === "/admin" || route === "/admin/audit") {
  show(route === "/admin" ? "Administration" : "Audit log");
} else if (route === "/orders/7") {
  show("Order 7");
} else if (route === "/throws") {
  const rows = undefined;
  show(rows.length);
} else if (route === "/empty") {
  // renders nothing
} else if (route === "/server-error") {
  show("Report");
  load("/api/report");
} else if (route === "/missing") {
  show("Customers");
  load("/api/customers");
} else if (route === "/slow") {
  show("Loading");
  load("/api/slow");
} else if (route === "/delayed") {
  setTimeout(() => show("Ready"), 1200);
} else if (route === "/items") {
  fetch("/api/items").then((response) => response.json()).then((items) => {
    const list = document.createElement("ul");
    for (const item of items) show(item.name, list);
    root.appendChild(list);
  });
} else if (route === "/portal") {
  const dialog = document.createElement("div");
  const heading = document.createElement("h2");
  heading.textContent = "Dialog page";
  dialog.appendChild(heading);
  document.body.appendChild(dialog);
} else if (route === "/profile") {
  fetch("/api/me").then((response) => (response.ok ? response.json() : null)).then((me) => {
    if (me) show(me.name);
  });
} else if (route === "/protected") {
  if (localStorage.getItem("token")) show("Account");
} else if (route === "/benign") {
  show("All good");
  setTimeout(() => { throw new Error("telemetry endpoint missing"); }, 0);
} else if (route === "/quote") {
  fetch("/api/quote").then((response) => show(response.ok ? "Quote" : "Quote unavailable"));
} else if (route === "/webgl") {
  const canvas = document.createElement("canvas");
  if (!(canvas.getContext("webgl2") || canvas.getContext("webgl"))) throw new Error("WebGL is not available");
  root.appendChild(canvas);
  show("3D view");
} else if (route === "/error-page") {
  const message = "rows is undefined";
  console.error(new TypeError(message));
  const title = document.createElement("h2");
  title.textContent = "Unexpected Application Error!";
  root.appendChild(title);
  const detail = document.createElement("h3");
  detail.textContent = message;
  root.appendChild(detail);
}
