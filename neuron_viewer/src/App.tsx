import React from "react";
import { useNavigate, Route, Routes, Link } from "react-router-dom";
import "./App.css";
import TransformerDebugger from "./TransformerDebugger/TransformerDebugger";
import { NextUIProvider } from "@nextui-org/react";
import Welcome from "./welcome";
import NodePage from "./nodePage";

const NotFoundPage: React.FC = () => {
  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-800">Page Not Found</h1>
        <p className="mt-4 text-xl text-gray-600">
          Sorry, the page you are looking for does not exist.
        </p>
        <Link
          to="/"
          className="mt-6 inline-block px-6 py-3 bg-blue-500 text-white font-medium text-lg leading-tight uppercase rounded shadow-md hover:bg-blue-700 hover:shadow-lg focus:bg-blue-700 focus:shadow-lg focus:outline-none focus:ring-0 active:bg-blue-800 active:shadow-lg transition duration-150 ease-in-out"
        >
          Go back home
        </Link>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const navigate = useNavigate();

  return (
    <NextUIProvider navigate={navigate}>
      <Routes>
        {/* Actual substantive pages */}
        <Route path="/" element={<Welcome />} />
        <Route path="/:model/:nodeTypeStr/:layerIndex/:nodeIndex" element={<NodePage />} />
        <Route path=":model/tdb_alpha" element={<TransformerDebugger />} />

        {/* Catch-all for bogus URLs */}
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </NextUIProvider>
  );
};

export default App;
