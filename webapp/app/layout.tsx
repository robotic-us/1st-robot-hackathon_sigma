import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SIGMA · cradle live",
  description: "Infant-responsive robotic cradle — live state",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
