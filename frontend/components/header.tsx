"use client"

import { useTheme } from "next-themes"
import Link from "next/link"
import Image from "next/image"
import { Moon, Sun, LogOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useAuth } from "@/lib/auth-context"

export function Header() {
  const { theme, setTheme } = useTheme()
  const { isAuthenticated, logout } = useAuth()

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 sm:h-24 md:h-28 flex items-center justify-between">
        {/* Logo only */}
        <Link href="/" className="flex items-center">
          <Image
            src="/biosight-logo.png"
            alt="BioSight Logo"
            width={160}
            height={160}
            className="object-contain"
          />
        </Link>

        {/* Navigation */}
        <nav className="hidden md:flex items-center gap-8">
          <Link href="/" className="text-sm font-medium text-foreground hover:text-primary transition-colors">
            Home
          </Link>
          <Link href="/dashboard" className="text-sm font-medium text-foreground hover:text-primary transition-colors">
            Dashboard
          </Link>
          <Link href="/upload" className="text-sm font-medium text-foreground hover:text-primary transition-colors">
            Upload
          </Link>
          <Link href="/docs" className="text-sm font-medium text-foreground hover:text-primary transition-colors">
            Knowledge Hub
          </Link>
        </nav>

        {/* Theme + Auth */}
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="h-9 w-9"
          >
            <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>
          {isAuthenticated ? (
            <Button variant="ghost" size="sm" onClick={logout} className="hidden sm:inline-flex gap-2">
              <LogOut className="h-4 w-4" />
              Logout
            </Button>
          ) : (
            <Button asChild variant="default" className="hidden sm:inline-flex">
              <Link href="/auth">Login</Link>
            </Button>
          )}
        </div>
      </div>
    </header>
  )
}
