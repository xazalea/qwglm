import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Qwen3-VL AI Assistant',
  description: 'Ultra-optimized 8B vision-language model with real-time inference',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet" />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}

// Suppress ESLint warning about custom fonts in layout
// eslint-disable-next-line @next/next/no-page-custom-font
