// Stand-in for a bundled single-page app: one behaviour per screen, chosen by the path.
const root = document.getElementById("root");
const route = location.pathname.replace(/\/+$/, "") || "/";

function show(text) {
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  root.appendChild(paragraph);
}

function load(path) {
  return fetch(path, { headers: { accept: "application/json" } })
    .then((response) => show(`${path}: ${response.status}`))
    .catch(() => show(`${path}: unreachable`));
}

if (route === "/") {
  show("Orders");
  const link = document.createElement("a");
  link.href = "/orders/7";
  link.textContent = "Order 7";
  root.appendChild(link);
  load("/api/orders");
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
}
