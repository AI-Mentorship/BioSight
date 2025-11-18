"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Footer } from "@/components/footer"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Upload, AlertCircle, CheckCircle } from "lucide-react"

export default function UploadPage() {
  const router = useRouter()
  const [isDragActive, setIsDragActive] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [patientId, setPatientId] = useState("")
  const [notes, setNotes] = useState("")
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [cancerType, setCancerType] = useState("")

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true)
    } else if (e.type === "dragleave") {
      setIsDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile.type.startsWith("image/")) {
        setFile(droppedFile)
      } else {
        alert("Please drop an image file")
      }
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000"

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault()

  if (!file || !patientId || !cancerType) {
    alert("Please select a cancer type, upload a file, and enter patient ID")
    return
  }

  setUploading(true)
  setUploadProgress(0)

  try {
    const formData = new FormData()
    formData.append("file", file)
    formData.append("cancer_type", cancerType)
    formData.append("patient_id", patientId)
    formData.append("notes", notes)

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.error || `Request failed with status ${res.status}`)
    }

    setUploadProgress(100)

    const data = await res.json()
    const caseId =
      data.case_id ||
      `CASE-${Math.random().toString(36).substr(2, 9).toUpperCase()}`

    // Build a query string with prediction info
    const query = new URLSearchParams({
      cancerType: data.cancer_type ?? cancerType,
      patientId: data.patient_id ?? patientId,
      predictedClass: data.predicted_class ?? "",
      confidence: data.confidence?.toString() ?? "",
      imageUrl: data.image_url ?? "",
    }).toString()

    router.push(`/analysis/${caseId}?${query}`)
  } catch (err: any) {
    console.error(err)
    alert(err.message || "Something went wrong while uploading")
  } finally {
    setUploading(false)
  }
}

  return (
    <>
      <main className="flex-1 px-4 sm:px-6 lg:px-8 py-8 bg-background">
        <div className="max-w-2xl mx-auto">

          <div className="mb-8">
            <h1 className="text-4xl sm:text-5xl font-serif font-bold text-[#054c5b] mb-2">Upload Image</h1>
            <p className="text-muted-foreground">Submit a histopathological image for AI analysis</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">

            {/* Cancer Type Dropdown */}
            <div className="space-y-2">
              <label htmlFor="cancer-type" className="text-sm font-medium">Cancer Type *</label>
              <select
                id="cancer-type"
                required
                value={cancerType}
                onChange={(e) => setCancerType(e.target.value)}
                className="w-full px-3 py-2 border border-input rounded-md bg-background text-foreground text-sm"
                disabled={uploading}
              >
                <option value="">-- Select Cancer Type --</option>
                <option value="lymphoma">Lymphoma</option>
                <option value="lung">Lung Cancer</option>
                <option value="breast">Breast Cancer</option>
                <option value="colon">Colon Cancer</option>
                <option value="cervical">Cervical Cancer</option>
                <option value="oral">Oral Cancer</option>
              </select>
            </div>

            {/* Dynamic Description */}
            {cancerType && (
              <p className="text-sm text-muted-foreground">
                Upload histopathology image for <span className="font-semibold">{cancerType}</span>.
              </p>
            )}

            {/* Drag and Drop Zone */}
            <Card
              className={`border-2 border-dashed transition-colors ${
                isDragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <CardContent className="pt-12 pb-12">
                <div className="text-center">
                  <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">Drag and drop your image here</h3>
                  <p className="text-muted-foreground mb-6">or click below to select from your computer</p>
                  <input type="file" id="file-input" accept="image/*" onChange={handleFileSelect} className="hidden" title="Select an image file" />
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => document.getElementById("file-input")?.click()}
                    disabled={uploading}
                  >
                    Select Image
                  </Button>
                </div>
              </CardContent>
            </Card>

            {file && (
              <Card className="border-border bg-secondary/30">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    <div className="flex-1">
                      <p className="font-medium text-foreground">{file.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {uploading && (
              <Card className="border-border">
                <CardContent className="pt-6">
                  <p className="text-sm font-medium mb-2">Uploading...</p>
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">{uploadProgress}%</p>
                </CardContent>
              </Card>
            )}

            {/* Patient ID */}
            <div className="space-y-2">
              <label htmlFor="patient-id" className="text-sm font-medium">Patient ID *</label>
              <Input
                id="patient-id"
                placeholder="e.g., PAT-2025-001"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                required
                disabled={uploading}
              />
            </div>

            {/* Notes */}
            <div className="space-y-2">
              <label htmlFor="notes" className="text-sm font-medium">Clinical Notes (Optional)</label>
              <textarea
                id="notes"
                placeholder="Add any relevant clinical context or observations..."
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                disabled={uploading}
                rows={4}
                className="w-full px-3 py-2 border border-input rounded-md bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50 text-sm"
              />
            </div>

            <Card className="border-border bg-secondary/20">
              <CardContent className="pt-4 pb-4">
                <div className="flex gap-3">
                  <AlertCircle className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                  <div className="text-sm text-foreground">
                    <p className="font-medium mb-1">Image Requirements</p>
                    <ul className="text-muted-foreground space-y-1">
                      <li>• Supported formats: JPG, PNG, TIFF</li>
                      <li>• Maximum file size: 50 MB</li>
                      <li>• Resolution: 400x400px minimum</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="flex gap-4">
              <Button
                type="submit"
                disabled={!file || !patientId || !cancerType || uploading}
                size="lg"
                className="font-semibold flex-1"
              >
                {uploading ? "Processing..." : "Submit for Analysis"}
              </Button>
              <Button type="button" variant="outline" size="lg" asChild>
                <Link href="/dashboard">Cancel</Link>
              </Button>
            </div>
          </form>
        </div>
      </main>
      <Footer />
    </>
  )
}
