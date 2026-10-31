import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import AnalysisPage from "./pages/Analysis";
import Home from "./pages/Home";
import ModelsPage from "./pages/Models";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="analysis/:id" element={<AnalysisPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
