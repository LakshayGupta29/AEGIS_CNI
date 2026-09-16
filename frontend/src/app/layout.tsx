import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Aegis-CNI | Cyber Resilience Intelligence",
  description:
    "AI-Powered Cyber Resilience Platform for Critical National Infrastructure — Autonomous behavioral anomaly detection, MITRE ATT&CK mapping, and SOAR orchestration",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} antialiased bg-[#060913] text-white min-h-screen`}
      >
        {children}
      </body>
    </html>
  );
}
