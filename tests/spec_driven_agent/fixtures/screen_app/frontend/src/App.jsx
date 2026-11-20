import { BrowserRouter, Route, Routes } from "react-router-dom";
import OrderList from "./pages/OrderList";
import OrderDetail from "./pages/OrderDetail";
import Broken from "./pages/Broken";
import Blank from "./pages/Blank";
import Report from "./pages/Report";
import CustomerList from "./pages/CustomerList";
import AdminLayout from "./pages/AdminLayout";
import AuditLog from "./pages/AuditLog";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<OrderList />} />
        <Route path="/orders/:id" element={<OrderDetail />} />
        <Route path="/throws" element={<Broken />} />
        <Route path="/empty" element={<Blank />} />
        <Route path="/server-error" element={<Report />} />
        <Route path="/missing" element={<CustomerList />} />
        <Route path="/admin" element={<AdminLayout />}>
          <Route path="audit" element={<AuditLog />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
