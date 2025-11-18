import "../styles/globals.css"
import { Header } from "@/components/header"
import { ThemeProvider } from "@/components/theme-provider"
import { AuthProvider } from "@/lib/auth-context"
import type { ReactNode } from "react"
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
          <AuthProvider>

            {/* Navbar always at top */}
            <Header />

            {/* Page content */}
            <main className="min-h-screen w-full">
              {children}
            </main>

          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
