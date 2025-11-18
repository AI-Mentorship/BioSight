"use client"

import type React from "react"
import { useState } from "react"
import { Footer } from "@/components/footer"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { useAuth } from "@/lib/auth-context"
import { useRouter } from "next/navigation"

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [role, setRole] = useState("oncologist")
  const [loading, setLoading] = useState(false)
  const { login, isAuthenticated } = useAuth()
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      if (isLogin) {
        await login(email, password)
        router.push("/dashboard")
      } else {
        await login(email, password)
        router.push("/dashboard")
      }
    } finally {
      setLoading(false)
    }
  }

  if (isAuthenticated) {
    router.push("/dashboard")
  }

  return (
    <>
      <main className="flex-1 flex items-center justify-center px-4 sm:px-6 lg:px-8 py-16 bg-secondary/20">
        <Card className="w-full max-w-md border-border">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl">{isLogin ? "Welcome Back" : "Create Account"}</CardTitle>
            <CardDescription>
              {isLogin ? "Sign in to access your dashboard" : "Join BioSight to begin analysis"}
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Email Input */}
              <div className="space-y-2">
                <label htmlFor="email" className="text-sm font-medium">
                  Email
                </label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>

              {/* Role Selection (Register only) */}
              {!isLogin && (
                <div className="space-y-2">
                  <label htmlFor="role" className="text-sm font-medium">
                    Role
                  </label>
                  <select
                    id="role"
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="w-full px-3 py-2 border border-input rounded-md bg-background text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  >
                    <option value="oncologist">Oncologist</option>
                    <option value="technician">Lab Technician</option>
                    <option value="admin">Administrator</option>
                  </select>
                </div>
              )}

              {/* Password Input */}
              <div className="space-y-2">
                <label htmlFor="password" className="text-sm font-medium">
                  Password
                </label>
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>

              {/* Submit Button */}
              <Button type="submit" disabled={loading} className="w-full font-semibold">
                {loading ? "Processing..." : isLogin ? "Sign In" : "Create Account"}
              </Button>

              {/* Toggle Mode */}
              <div className="text-center text-sm">
                <span className="text-muted-foreground">
                  {isLogin ? "Don't have an account? " : "Already have an account? "}
                </span>
                <button
                  type="button"
                  onClick={() => setIsLogin(!isLogin)}
                  className="text-primary hover:underline font-medium"
                >
                  {isLogin ? "Sign up" : "Sign in"}
                </button>
              </div>
            </form>

            {/* Info Box */}
            <div className="mt-6 p-4 rounded-lg bg-secondary/50 border border-border">
              <p className="text-xs text-muted-foreground">
                <strong>Demo Credentials:</strong> Use any email/password combination to test. In production, this would
                integrate with Supabase for secure authentication.
              </p>
            </div>
          </CardContent>
        </Card>
      </main>
      <Footer />
    </>
  )
}
