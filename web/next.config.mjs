// spec §3, §4.2 — fully static export in production, no SSR/Node at runtime.
// `next dev` only: proxy /api + /runtime-config.json + /media to the backend
// (pixi run app:dev), matching the spec's "Next.js dev server :3000 (proxied)".
const isDev = process.env.NODE_ENV === "development";
const backend = process.env.SPOTTER_BACKEND_URL || "http://127.0.0.1:8756";

/** @type {import('next').NextConfig} */
const nextConfig = {
  ...(isDev ? {} : { output: "export" }),
  reactStrictMode: true,
  images: { unoptimized: true },
  trailingSlash: false,
  ...(isDev && {
    async rewrites() {
      return [
        { source: "/api/:path*", destination: `${backend}/api/:path*` },
        { source: "/media/:path*", destination: `${backend}/media/:path*` },
        { source: "/runtime-config.json", destination: `${backend}/runtime-config.json` },
      ];
    },
  }),
};

export default nextConfig;
