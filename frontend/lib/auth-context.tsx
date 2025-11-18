"use client"

import type React from "react"

import { createContext, useContext, useEffect, useState } from "react"

interface AuthContextType {
  isAuthenticated: boolean
  user: { email: string; role: string } | null
  login: (email: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState<{ email: string; role: string } | null>(null)

  useEffect(() => {
    // Load auth state from localStorage on mount
    const stored = localStorage.getItem("biosight_auth")
    if (stored) {
      const auth = JSON.parse(stored)
      setIsAuthenticated(true)
      setUser(auth)
    }
  }, [])

  const login = async (email: string, password: string) => {
    // Simulate auth delay
    await new Promise((resolve) => setTimeout(resolve, 500))
    const userData = { email, role: "oncologist" }
    setIsAuthenticated(true)
    setUser(userData)
    localStorage.setItem("biosight_auth", JSON.stringify(userData))
  }

  const logout = () => {
    setIsAuthenticated(false)
    setUser(null)
    localStorage.removeItem("biosight_auth")
  }

  return <AuthContext.Provider value={{ isAuthenticated, user, login, logout }}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider")
  }
  return context
}
