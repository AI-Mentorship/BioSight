"use client"

import { useState } from "react"
import Link from "next/link"
import { Footer } from "@/components/footer"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Eye, Download, Trash2, Plus, Search } from "lucide-react"
import { ProtectedRoute } from "@/components/protected-route"

// Mock data for cases
const MOCK_CASES = [
  {
    id: "CASE-001",
    patientId: "PAT-2025-001",
    status: "completed",
    confidence: 94,
    date: "2025-01-10",
    diagnosis: "Lymphoma",
  },
  {
    id: "CASE-002",
    patientId: "PAT-2025-002",
    status: "completed",
    confidence: 87,
    date: "2025-01-09",
    diagnosis: "Breast Cancer",
  },
  {
    id: "CASE-003",
    patientId: "PAT-2025-003",
    status: "processing",
    confidence: 0,
    date: "2025-01-08",
    diagnosis: "Analysis in Progress",
  },
  {
    id: "CASE-004",
    patientId: "PAT-2025-004",
    status: "completed",
    confidence: 91,
    date: "2025-01-07",
    diagnosis: "Lung Cancer",
  },
  {
    id: "CASE-005",
    patientId: "PAT-2025-005",
    status: "completed",
    confidence: 89,
    date: "2025-01-06",
    diagnosis: "Colon Cancer",
  },
]

export default function DashboardPage() {
  const [searchTerm, setSearchTerm] = useState("")
  const [cases, setCases] = useState(MOCK_CASES)

  const filteredCases = cases.filter(
    (c) =>
      c.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      c.patientId.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  const handleDelete = (id: string) => {
    setCases(cases.filter((c) => c.id !== id))
  }

  const getStatusBadge = (status: string) => {
    if (status === "completed") {
      return (
        <span className="px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
          Completed
        </span>
      )
    }
    return (
      <span className="px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
        Processing
      </span>
    )
  }

  const dashboardContent = (
    <>
      <main className="flex-1 px-4 sm:px-6 lg:px-8 py-8 bg-background">
        <div className="max-w-7xl mx-auto">
          {/* Page Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
            <div>
              <h1 className="text-4xl sm:text-5xl font-serif font-bold text-[#054c5b] mb-2">Cases</h1>
              <p className="text-muted-foreground">Manage and review analyzed histopathological cases</p>
            </div>
            <Button asChild size="lg" className="font-semibold">
              <Link href="/upload">
                <Plus className="w-4 h-4 mr-2" />
                New Case
              </Link>
            </Button>
          </div>

          {/* Search Bar */}
          <Card className="mb-6 border-border">
            <CardContent className="pt-6">
              <div className="relative">
                <Search className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search by Case ID or Patient ID..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </CardContent>
          </Card>

          {/* Cases Table */}
          <Card className="border-border overflow-hidden">
            <CardHeader>
              <CardTitle>Recent Cases</CardTitle>
              <CardDescription>Total: {cases.length} cases</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border bg-secondary/50">
                      <th className="text-left py-4 px-4 font-semibold text-foreground">Case ID</th>
                      <th className="text-left py-4 px-4 font-semibold text-foreground">Patient ID</th>
                      <th className="text-left py-4 px-4 font-semibold text-foreground">Diagnosis</th>
                      <th className="text-left py-4 px-4 font-semibold text-foreground">Status</th>
                      <th className="text-left py-4 px-4 font-semibold text-foreground">Confidence</th>
                      <th className="text-left py-4 px-4 font-semibold text-foreground">Date</th>
                      <th className="text-left py-4 px-4 font-semibold text-foreground">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredCases.length > 0 ? (
                      filteredCases.map((caseItem) => (
                        <tr
                          key={caseItem.id}
                          className="border-b border-border hover:bg-secondary/30 transition-colors"
                        >
                          <td className="py-4 px-4 font-mono text-primary">{caseItem.id}</td>
                          <td className="py-4 px-4">{caseItem.patientId}</td>
                          <td className="py-4 px-4">{caseItem.diagnosis}</td>
                          <td className="py-4 px-4">{getStatusBadge(caseItem.status)}</td>
                          <td className="py-4 px-4">
                            {caseItem.status === "completed" ? (
                              <span className="font-semibold text-foreground">{caseItem.confidence}%</span>
                            ) : (
                              <span className="text-muted-foreground">—</span>
                            )}
                          </td>
                          <td className="py-4 px-4 text-muted-foreground">{caseItem.date}</td>
                          <td className="py-4 px-4">
                            <div className="flex items-center gap-2">
                              {caseItem.status === "completed" && (
                                <>
                                  <Button asChild variant="ghost" size="sm" className="h-8 w-8 p-0">
                                    <Link href={`/analysis/${caseItem.id}`} title="View analysis">
                                      <Eye className="w-4 h-4" />
                                    </Link>
                                  </Button>
                                  <Button asChild variant="ghost" size="sm" className="h-8 w-8 p-0">
                                    <Link href={`/report/${caseItem.id}`} title="Generate report">
                                      <Download className="w-4 h-4" />
                                    </Link>
                                  </Button>
                                </>
                              )}
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                                onClick={() => handleDelete(caseItem.id)}
                                title="Delete case"
                              >
                                <Trash2 className="w-4 h-4" />
                              </Button>
                            </div>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={7} className="py-8 px-4 text-center text-muted-foreground">
                          No cases found
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
      <Footer />
    </>
  )

  return <ProtectedRoute>{dashboardContent}</ProtectedRoute>
}
