"use client"

import { useState } from "react"
import { useSearchParams } from "next/navigation"
import Link from "next/link"
import { Footer } from "@/components/footer"
import { Button } from "@/components/ui/button"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { ChevronDown } from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"

const confidenceData = [
  { region: "Region A", confidence: 92 },
  { region: "Region B", confidence: 87 },
  { region: "Region C", confidence: 94 },
  { region: "Region D", confidence: 85 },
]

export default function AnalysisPage({ params }: { params: { caseId: string } }) {
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [expandedRegion, setExpandedRegion] = useState<string | null>(null)

  // 🔹 Read query params from URL
  const searchParams = useSearchParams()
  const cancerType = searchParams.get("cancerType") || "Unknown"
  const patientId = searchParams.get("patientId") || "Unknown"
  const predictedClass = searchParams.get("predictedClass") || "N/A"
  const confidenceStr = searchParams.get("confidence")
  const imageUrl = searchParams.get("imageUrl") || "/placeholder.svg?key=analysis1"

  // backend sends confidence as 0–1, convert to %
  const confidencePercent =
    confidenceStr != null && confidenceStr !== ""
      ? (parseFloat(confidenceStr) * 100).toFixed(1)
      : "N/A"

  return (
    <>
      <main className="flex-1 px-4 sm:px-6 lg:px-8 py-8 bg-background">
        <div className="max-w-7xl mx-auto">
          {/* Page Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h1 className="text-3xl font-bold text-foreground">
                  Analysis Report
                </h1>
                <p className="text-muted-foreground">{params.caseId}</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Patient: <span className="font-medium">{patientId}</span> ·{" "}
                  Cancer type:{" "}
                  <span className="font-medium capitalize">
                    {cancerType}
                  </span>
                </p>
              </div>
              <div className="flex gap-3">
                <Button variant="outline" asChild>
                  <Link href="/dashboard">Back to Dashboard</Link>
                </Button>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Image Section */}
            <div className="lg:col-span-2">
              <Card className="border-border overflow-hidden">
                <CardHeader>
                  <CardTitle>Histopathological Image</CardTitle>
                </CardHeader>
                <CardContent>
                  {/* Image Viewer */}
                  <div className="bg-secondary/50 rounded-lg overflow-hidden border border-border relative">
                    <img
                      src={imageUrl}
                      alt="Histopathological image for analysis"
                      className="w-full h-auto"
                    />
                    {showHeatmap && (
                      <div className="absolute inset-0 bg-gradient-to-br from-transparent via-accent/20 to-primary/20 pointer-events-none rounded-lg" />
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
            {/* Right Sidebar */}
            <div className="space-y-6">
              {/* Summary Stats */}
              <Card className="border-border">
                <CardHeader>
                  <CardTitle className="text-lg">Summary</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">
                      Overall Confidence
                    </p>
                    <p className="text-3xl font-bold text-primary">
                      {confidencePercent !== "N/A"
                        ? `${confidencePercent}%`
                        : "N/A"}
                    </p>
                  </div>
                  <div className="h-px bg-border" />
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">
                      Primary Diagnosis
                    </p>
                    <p className="font-semibold text-foreground">
                      {predictedClass}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">
                      Cancer Type
                    </p>
                    <p className="font-semibold text-foreground capitalize">
                      {cancerType}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">
                      Analysis Date
                    </p>
                    <p className="font-semibold text-foreground">
                      {/* you can replace this with real date from backend later */}
                      {new Date().toISOString().slice(0, 10)}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </>
  )
}