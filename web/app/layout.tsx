import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { Chrome } from "@/components/Chrome";

export const metadata: Metadata = {
  title: "Salamander Spotter",
  description: "Fire-salamander census for Kibbutz Sasa",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap"
        />
      </head>
      <body>
        <Providers>
          <Chrome>{children}</Chrome>
        </Providers>
      </body>
    </html>
  );
}
