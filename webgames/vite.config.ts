import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import CloudflarePagesFunctions from "vite-plugin-cloudflare-functions";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), CloudflarePagesFunctions()],
});
