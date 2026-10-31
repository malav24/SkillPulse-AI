/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.VITE_PROXY_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: apiTarget, changeOrigin: true } },
  },
  build: {
    chunkSizeWarningLimit: 5000, // Plotly is large; it is loaded lazily only on the results page
    rollupOptions: {
      output: { manualChunks: { vendor: ["react", "react-dom", "react-router-dom"] } },
    },
  },
  preview: { port: 4173, proxy: { "/api": { target: apiTarget, changeOrigin: true } } },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
