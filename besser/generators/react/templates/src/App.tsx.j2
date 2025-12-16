import React from "react";
import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import { TableProvider } from "./contexts/TableContext";
import componentsData from "./data/ui_components.json";

function App() {
  // Dynamically extract all routes from pages
  const pages = componentsData?.pages ?? [];
  const routes = pages
    .filter((page: any) => page.route_path)
    .map((page: any) => page.route_path);

  return (
    <TableProvider>
      <div className="app-container">
        {/* Page Routes */}
        <main className="app-main">
          <Routes>
            {/* Dynamically generate routes from JSON */}
            {routes.map((route: string) => (
              <Route key={route} path={route} element={<Home />} />
            ))}
            {/* Redirect root to first page or home */}
            <Route path="/" element={<Home />} />
          </Routes>
        </main>
      </div>
    </TableProvider>
  );
}export default App;