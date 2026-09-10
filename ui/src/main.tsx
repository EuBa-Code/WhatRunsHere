/** Mounts the window. */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./theme.css";

const root = document.getElementById("root");
if (!root) throw new Error("the window has no root element to mount into");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
