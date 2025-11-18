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

                  {/* Controls */}
                  <div className="flex gap-2 mt-4">
                    <Button
                      variant={showHeatmap ? "default" : "outline"}
                      onClick={() => setShowHeatmap(!showHeatmap)}
                      size="sm"
                    >
                      {showHeatmap ? "Hide" : "Show"} Heatmap
                    </Button>
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

              {/* Explanation */}
              <Card className="border-border">
                <CardHeader>
                  <CardTitle className="text-lg">Why?</CardTitle>
                  <CardDescription>Model interpretability</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-foreground leading-relaxed">
                    The model&apos;s prediction reflects patterns learned from
                    histopathological images of{" "}
                    <span className="font-semibold capitalize">
                      {cancerType}
                    </span>
                    . This includes cellular morphology, tissue architecture,
                    and texture features associated with the predicted class{" "}
                    <span className="font-semibold">{predictedClass}</span>.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Confidence Breakdown */}
          <Card className="border-border mt-6">
            <CardHeader>
              <CardTitle>Confidence Breakdown by Region</CardTitle>
              <CardDescription>
                AI model confidence scores across analyzed tissue regions
                (placeholder demo)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={confidenceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis dataKey="region" stroke="var(--color-muted-foreground)" />
                  <YAxis stroke="var(--color-muted-foreground)" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--color-card)",
                      border: "1px solid var(--color-border)",
                      borderRadius: "var(--radius)",
                    }}
                  />
                  <Bar
                    dataKey="confidence"
                    fill="var(--color-primary)"
                    radius={[8, 8, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Region Details */}
          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-4 text-foreground">
              Region Details
            </h3>
            <div className="space-y-3">
              {confidenceData.map((region) => (
                <Card key={region.region} className="border-border">
                  <button
                    onClick={() =>
                      setExpandedRegion(
                        expandedRegion === region.region ? null : region.region
                      )
                    }
                    className="w-full px-6 py-4 flex items-center justify-between hover:bg-secondary/30 transition-colors"
                  >
                    <div className="flex items-center gap-4 flex-1">
                      <div className="w-3 h-3 rounded-full bg-primary" />
                      <div className="text-left">
                        <p className="font-semibold text-foreground">
                          {region.region}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Confidence: {region.confidence}%
                        </p>
                      </div>
                    </div>
                    <ChevronDown
                      className={`w-5 h-5 text-muted-foreground transition-transform ${
                        expandedRegion === region.region ? "rotate-180" : ""
                      }`}
                    />
                  </button>

                  {expandedRegion === region.region && (
                    <div className="px-6 py-4 border-t border-border bg-secondary/20">
                      <p className="text-sm text-foreground">
                        Tissue analysis shows characteristic cellular morphology
                        associated with the model&apos;s prediction. This
                        section is placeholder text you can later replace with
                        region-level explanations from your heatmap pipeline.
                      </p>
                    </div>
                  )}
                </Card>
              ))}
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </>
  )
}