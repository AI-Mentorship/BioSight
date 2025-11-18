"use client"

import { useState } from "react"
import Link from "next/link"
import { Footer } from "@/components/footer"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Search, ChevronRight, BookOpen } from "lucide-react"

const sections = [
  {
    id: "getting-started",
    title: "Getting Started",
    items: [
      { title: "Creating Your Account", slug: "account-setup" },
      { title: "Dashboard Overview", slug: "dashboard-guide" },
      { title: "Your First Analysis", slug: "first-analysis" },
    ],
  },
  {
    id: "technical",
    title: "Technical Details",
    items: [
      { title: "How the AI Model Works", slug: "model-architecture" },
      { title: "Image Processing Pipeline", slug: "image-processing" },
      { title: "Model Accuracy Metrics", slug: "accuracy-metrics" },
      { title: "Supported Image Formats", slug: "image-formats" },
    ],
  },
  {
    id: "clinical",
    title: "Clinical & Ethics",
    items: [
      { title: "Ethical Limitations", slug: "ethical-guidelines" },
      { title: "Confidence Scores Explained", slug: "confidence-explanation" },
      { title: "Clinical Validation Studies", slug: "validation-studies" },
      { title: "Regulatory Compliance", slug: "compliance" },
    ],
  },
  {
    id: "faq",
    title: "FAQ & Troubleshooting",
    items: [
      { title: "Frequently Asked Questions", slug: "faq" },
      { title: "Upload Issues", slug: "upload-troubleshooting" },
      { title: "Interpretation Guide", slug: "interpretation-guide" },
      { title: "Support & Contact", slug: "support" },
    ],
  },
]

export default function DocsPage() {
  const [searchTerm, setSearchTerm] = useState("")

  const filteredSections = sections
    .map((section) => ({
      ...section,
      items: section.items.filter((item) => item.title.toLowerCase().includes(searchTerm.toLowerCase())),
    }))
    .filter((section) => section.items.length > 0)

  return (
    <>
      <main className="flex-1 px-4 sm:px-6 lg:px-8 py-8 bg-background">
        <div className="max-w-6xl mx-auto">
          {/* Page Header */}
          <div className="mb-12 text-center">
            <h1 className="text-4xl sm:text-5xl font-serif font-bold text-[#054c5b] mb-4">Knowledge Hub</h1>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Learn how to use BioSight, understand our AI model, and explore ethical considerations for clinical
              deployment
            </p>
          </div>

          {/* Search */}
          <div className="max-w-2xl mx-auto mb-12">
            <div className="relative">
              <Search className="absolute left-3 top-3 w-5 h-5 text-muted-foreground" />
              <Input
                placeholder="Search documentation..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 py-6 text-base"
              />
            </div>
          </div>

          {/* Documentation Sections */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            {filteredSections.map((section) => (
              <Card key={section.id} className="border-border hover:shadow-md transition-shadow">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BookOpen className="w-5 h-5 text-primary" />
                    {section.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-3">
                    {section.items.map((item) => (
                      <li key={item.slug}>
                        <Link
                          href={`#${item.slug}`}
                          className="flex items-center gap-2 text-foreground hover:text-primary transition-colors group"
                        >
                          <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                          {item.title}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Featured Articles */}
          <div className="mb-12">
            <h2 className="text-2xl font-bold text-foreground mb-6">Featured Articles</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                {
                  title: "How the AI Model Works",
                  description:
                    "Deep dive into the neural network architecture and training methodology behind BioSight",
                  icon: "🧠",
                },
                {
                  title: "Accuracy Metrics",
                  description: "Understanding sensitivity, specificity, and ROC-AUC scores in clinical deployment",
                  icon: "📊",
                },
                {
                  title: "Ethical Guidelines",
                  description: "Important limitations and responsible use of AI in clinical diagnostics",
                  icon: "⚖️",
                },
              ].map((article, i) => (
                <Card key={i} className="border-border hover:shadow-md transition-shadow cursor-pointer">
                  <CardHeader>
                    <CardTitle className="text-lg">{article.title}</CardTitle>
                    <CardDescription>{article.description}</CardDescription>
                  </CardHeader>
                </Card>
              ))}
            </div>
          </div>

          {/* Content Sections (Placeholder) */}
          <div className="space-y-12">
            <section id="model-architecture" className="scroll-mt-8">
              <h2 className="text-2xl font-bold text-foreground mb-4">How the AI Model Works</h2>
              <Card className="border-border">
                <CardContent className="pt-6">
                  <div className="space-y-4 text-foreground">
                    <p>
                      BioSight uses a deep convolutional neural network trained on 50,000+ annotated histopathological
                      images. The model architecture includes:
                    </p>
                    <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                      <li>Multi-scale feature extraction for detecting tissue patterns at different magnifications</li>
                      <li>Attention mechanisms to focus on diagnostically relevant regions</li>
                      <li>Ensemble predictions combining multiple model checkpoints for robust confidence scoring</li>
                      <li>Grad-CAM visualization to highlight regions influencing the diagnosis</li>
                    </ul>
                    <p className="text-sm text-muted-foreground italic">
                      The model is updated regularly with new clinical data to improve accuracy and expand supported
                      cancer subtypes.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </section>

            <section id="ethical-guidelines">
              <h2 className="text-2xl font-bold text-foreground mb-4">Ethical Considerations & Limitations</h2>
              <Card className="border-border">
                <CardContent className="pt-6">
                  <div className="space-y-4 text-foreground">
                    <div className="bg-yellow-50 dark:bg-yellow-950/30 p-4 rounded-lg border border-yellow-200 dark:border-yellow-900">
                      <p className="font-semibold text-yellow-900 dark:text-yellow-100 mb-2">Critical Limitations</p>
                      <ul className="list-disc list-inside space-y-2 text-yellow-800 dark:text-yellow-200 text-sm">
                        <li>AI predictions are probabilistic estimates, not definitive diagnoses</li>
                        <li>Model performance depends heavily on image quality and preparation standards</li>
                        <li>Bias in training data may affect diagnostic accuracy across patient demographics</li>
                        <li>Clinical judgment and human pathologist review are essential for all diagnoses</li>
                      </ul>
                    </div>
                    <p className="text-sm">
                      BioSight is designed as a decision-support tool to augment clinical workflows, not replace expert
                      pathologists. All diagnoses made using BioSight must be independently verified by qualified
                      clinicians.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </section>

            <section id="faq">
              <h2 className="text-2xl font-bold text-foreground mb-4">Frequently Asked Questions</h2>
              <div className="space-y-4">
                {[
                  {
                    q: "What image formats does BioSight support?",
                    a: "BioSight supports JPEG, PNG, and TIFF formats. Minimum resolution is 400x400 pixels; recommended resolution is 1024x1024 or higher.",
                  },
                  {
                    q: "How long does analysis typically take?",
                    a: "Most images are analyzed within 30-60 seconds. Complex images may take 2-3 minutes.",
                  },
                  {
                    q: "Can I download my analysis reports?",
                    a: "Yes, you can export reports as PDF with full clinical data and interpretability explanations.",
                  },
                  {
                    q: "Is my patient data secure?",
                    a: "All data is encrypted in transit and at rest. We comply with HIPAA and GDPR regulations.",
                  },
                ].map((faq, i) => (
                  <Card key={i} className="border-border">
                    <CardHeader>
                      <CardTitle className="text-base">{faq.q}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-foreground">{faq.a}</p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </section>
          </div>
        </div>
      </main>
      <Footer />
    </>
  )
}
