import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import SidebarNav from "./sidebar-nav";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AetherShield | Real-Time Deepfake Detection Platform",
  description: "Enterprise-grade real-time AI media forensics and authentication pipeline",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                const saved = localStorage.getItem('aethershield_theme');
                const theme = saved || 'light';
                document.documentElement.classList.remove('light', 'dark');
                document.documentElement.classList.add(theme);
              } catch (_) {}
            `
          }}
        />
      </head>
      <body className="min-h-full flex flex-row overflow-hidden bg-slate-50 text-slate-900">
        <SidebarNav />
        <div className="flex-1 flex flex-col overflow-y-auto relative scanline h-screen font-sans">
          {children}
        </div>
      </body>
    </html>
  );
}
