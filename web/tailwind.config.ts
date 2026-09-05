import type { Config } from "tailwindcss";

// spec §13 — the canvas design system.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ground: "#faf8f5",
        ink: "#33302c",
        panel: "#ffffff",
        sidebar: "#2b2825",
        accent: "#b9782a",
        confirm: "#2f7d4f",
        warn: "#8a5a1e",
        info: "#3f5f9c",
        danger: "#a5443b",
      },
      fontFamily: {
        head: ['"Hanken Grotesk"', "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        card: "0 1px 3px rgba(51,48,44,0.08), 0 6px 20px rgba(51,48,44,0.05)",
      },
    },
  },
  plugins: [],
};

export default config;
