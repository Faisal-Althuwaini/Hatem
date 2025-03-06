import HomePage from "./pages/HomePage";
import Layout from "./components/Layout";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import ChatBotPage from "./pages/ChatBotPage";
import CalculatorPage from "./pages/CalculatorPage";
import GpaPage from "./pages/GpaPage";
import ResourcesPage from "./pages/ResourcesPage";

function App() {
  return (
    <BrowserRouter>
      <div className="">
        <Layout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/chatbot" element={<ChatBotPage />} />
            <Route path="/calculator" element={<CalculatorPage />} />
            <Route path="/gpacalc" element={<GpaPage />} />
            <Route path="/resources" element={<ResourcesPage />} />
          </Routes>
        </Layout>
      </div>
    </BrowserRouter>
  );
}

export default App;
