"use client"

import { useState, useRef } from "react"
import { Upload, Check, AlertCircle } from "lucide-react"

export default function ImageUpload() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [prediction, setPrediction] = useState<{ result: string; confidence: number } | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [selectedModel, setSelectedModel] = useState("oral-cancer")

  const models = [
    { value: "oral-cancer", label: "Oral Cancer" },
    { value: "breast-cancer", label: "Breast Cancer" },
    { value: "lung-cancer", label: "Lung Cancer" },
    { value: "lymphoma", label: "Lymphoma" },
    { value: "cervical-cancer", label: "Cervical Cancer" },
    { value: "colon-cancer", label: "Colon Cancer" },
  ]

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setSelectedFile(file)
    const reader = new FileReader()
    reader.onloadend = () => setUploadedImage(reader.result as string)
    reader.readAsDataURL(file)
    setPrediction(null)
  }

  // Handle prediction
  const handleAnalyze = async () => {
    if (!selectedFile) return
    setIsAnalyzing(true)

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)
      formData.append("model_name", selectedModel)

      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      })

      const data = await res.json()
      if (data.error) {
        alert(data.error)
      } else {
        setPrediction({
          result: data.prediction,
          confidence: data.confidence,
        })
      }
    } catch (err) {
      console.error(err)
      alert("Failed to get prediction from backend.")
    }

    setIsAnalyzing(false)
  }

  const handleReset = () => {
    setUploadedImage(null)
    setSelectedFile(null)
    setPrediction(null)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  return (
    <section id="upload" className="py-20 lg:py-32 bg-gradient-to-br from-primary/5 to-background">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-4">Upload Image for Analysis</h2>
          <p className="text-xl text-muted-foreground">Upload a tissue sample image to get an instant AI-powered prediction</p>
        </div>

        <div className="bg-card border border-border rounded-2xl p-8">
          {/* Model Selection */}
          <div className="mb-8">
            <label className="block text-sm font-semibold text-foreground mb-4">Select Detection Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-4 py-3 bg-background border border-border rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              {models.map((model) => (
                <option key={model.value} value={model.value}>
                  {model.label}
                </option>
              ))}
            </select>
          </div>

          {/* Upload Area */}
          {!uploadedImage ? (
            <div
              onClick={() => fileInputRef.current?.click()}
              className="relative border-2 border-dashed border-border rounded-xl p-12 bg-background/50 hover:bg-background transition-colors cursor-pointer group"
            >
              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
              <div className="flex flex-col items-center justify-center gap-4 text-center">
                <div className="p-3 bg-primary/10 rounded-lg group-hover:bg-primary/20 transition-colors">
                  <Upload className="w-8 h-8 text-primary" />
                </div>
                <div>
                  <p className="text-lg font-semibold text-foreground">Click to upload or drag and drop</p>
                  <p className="text-sm text-muted-foreground mt-1">PNG, JPG, GIF up to 10MB</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="rounded-xl overflow-hidden border border-border">
                <img src={uploadedImage} alt="Uploaded tissue sample" className="w-full h-64 object-cover" />
              </div>

              {prediction ? (
                <div
                  className={`p-6 rounded-xl border-2 ${
                    prediction.result === "Cancer Detected"
                      ? "border-destructive/30 bg-destructive/5"
                      : "border-green-500/30 bg-green-500/5"
                  }`}
                >
                  <div className="flex items-start gap-4">
                    <div className="mt-1">
                      {prediction.result === "Cancer Detected" ? (
                        <AlertCircle className="w-6 h-6 text-destructive" />
                      ) : (
                        <Check className="w-6 h-6 text-green-500" />
                      )}
                    </div>
                    <div className="flex-1">
                      <h3
                        className={`text-lg font-bold ${
                          prediction.result === "Cancer Detected" ? "text-destructive" : "text-green-500"
                        }`}
                      >
                        {prediction.result}
                      </h3>
                      <p className="text-muted-foreground mt-1">
                        Model confidence:{" "}
                        <span className="font-semibold text-foreground">{prediction.confidence.toFixed(2)}%</span>
                      </p>
                      <p className="text-sm text-muted-foreground mt-2">
                        This prediction should be validated by a medical professional.
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="w-full px-6 py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
                >
                  {isAnalyzing ? "Analyzing..." : "Analyze Image"}
                </button>
              )}

              <button
                onClick={handleReset}
                className="w-full px-6 py-3 border border-border text-foreground rounded-lg font-semibold hover:bg-muted transition-colors"
              >
                Upload Different Image
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
