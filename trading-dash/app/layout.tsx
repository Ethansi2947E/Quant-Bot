import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import type React from "react" // Import React

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Trading Bot Dashboard",
  description: "Real-time trading bot monitoring and control dashboard",
  viewport: "width=device-width, initial-scale=1",
  themeColor: "#000000",
  icons: {
    icon: "/favicon.ico",
    apple: "/apple-touch-icon.png",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="h-full">
      <body
        className={`${inter.className} antialiased bg-[var(--background-primary)] text-[var(--text-primary)] h-full`}
      >
        <div className="min-h-screen">{children}</div>
      </body>
    </html>
  )
}

