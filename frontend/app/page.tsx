import Link from "next/link"
import { Footer } from "@/components/footer"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowRight, Microscope, BarChart3, FileText, Lock, ShieldCheck, Users, Image as ImageIcon } from "lucide-react"
import Carousel from '@/components/carousel'
import ScrollRevealSection from '@/components/scroll-reveal-section'
import Image from "next/image"

export default function Home() {
  return (
    <main className="flex flex-col min-h-screen">

      {/* HERO SECTION */}
      <section className="relative min-h-[80vh] flex flex-col items-center justify-center px-6 bg-[#f6fbfc]">

        {/* Headline */}
        <h1
          className="text-[52px] sm:text-[70px] font-serif font-bold leading-tight text-center text-[#054c5b]
          opacity-0 animate-fadeSlideUp"
        >
          Redefining <span className="italic">Vision</span>
          <br />
          in <span className="italic">Diagnostics</span>
        </h1>

        {/* Subheadline */}
        <p className="mt-6 text-lg text-gray-600 max-w-2xl text-center opacity-0 animate-fadeSlideUp delay-200">
          BioSight assists oncologists with AI-powered analysis of histopathological images to accelerate clinical decision-making.
        </p>

        {/* CTA */}
        <div className="mt-10 opacity-0 animate-fadeSlideUp delay-300">
          <Link
            href="/auth"
            className="px-10 py-4 rounded-full text-white font-semibold text-lg shadow-md"
            style={{ backgroundColor: "#054c5b" }}
          >
            Get Started
          </Link>
        </div>

      </section>

      {/* WHY BIOSIGHT */}
      <ScrollRevealSection>
        <section className="py-12 px-4 sm:px-6 lg:px-8 bg-background">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-3xl sm:text-4xl font-serif font-bold mb-2 text-[#054c5b]">Why BioSight?</h2>
              <p className="text-muted-foreground">Built for clinicians, explainability, and privacy - all at first deployment.</p>
            </div>

            <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-6">
              <div className="flex flex-col items-start gap-4 p-6 bg-card rounded-lg border border-border">
                <div className="w-12 h-12 rounded-md bg-primary/10 flex items-center justify-center">
                  <Microscope className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold">Clinician-Focused</h3>
                <p className="text-sm text-muted-foreground">Built to match the way clinicians actually work, with a straightforward flow and results that are easy to review.</p>
              </div>

              <div className="flex flex-col items-start gap-4 p-6 bg-card rounded-lg border border-border">
                <div className="w-12 h-12 rounded-md bg-accent/10 flex items-center justify-center">
                  <ShieldCheck className="w-6 h-6 text-accent" />
                </div>
                <h3 className="text-lg font-semibold">Clear & Interpretable</h3>
                <p className="text-sm text-muted-foreground">Visual heatmaps and confidence scores to support interpretability.</p>
              </div>

              <div className="flex flex-col items-start gap-4 p-6 bg-card rounded-lg border border-border">
                <div className="w-12 h-12 rounded-md bg-primary/10 flex items-center justify-center">
                  <Lock className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold">Secure & Private</h3>
                <p className="text-sm text-muted-foreground">Uses careful data handling practices and access controls to help keep patient information protected.</p>
              </div>
            </div>
          </div>
        </section>
      </ScrollRevealSection>

      {/* SAMPLE ANALYSIS PREVIEW */}
      <ScrollRevealSection>
        <section className="py-12 px-4 sm:px-6 lg:px-8 bg-background">
          <div className="max-w-6xl mx-auto">
            <h3 className="text-2xl font-serif font-bold mb-6 text-[#054c5b]">Sample Analysis Preview</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div className="rounded-lg overflow-hidden border border-border bg-card p-4 flex flex-col items-center justify-center">
                <div className="w-full h-64 dark:bg-gray-800 rounded-md flex items-center justify-center">
                  <div className="w-full h-64 rounded-md overflow-hidden relative">
                    <div className="w-full max-w-[384px] h-auto mx-auto">
                      <Image
                        src="/lungaca.png"
                        alt="Input Slide"
                        width={384}
                        height={384}
                        className="object-contain"
                      />
                    </div>
                  </div>
                </div>
                <p className="mt-3 text-sm text-muted-foreground">Lung Adenocarcinoma</p>
              </div>
              <div className="rounded-lg overflow-hidden border border-border bg-card p-4 flex flex-col items-center justify-center">
                <div className="w-full h-64 rounded-md overflow-hidden relative">
                  <Image
                    src="/heatmaplung.png"
                    alt="Input Slide"
                    fill
                    className="rounded-md object-contain"
                  />
                </div>
                <p className="mt-3 text-sm text-muted-foreground">Heatmap for Lung Adenocarcinoma</p>
              </div>
            </div>
          </div>
        </section>
      </ScrollRevealSection>

      {/* OUR MISSION */}
      <ScrollRevealSection>
        <section className="py-8 px-4 sm:px-6 lg:px-8">
          <div className="max-w-4xl mx-auto text-center">
            <h3 className="text-2xl font-serif font-bold mb-3 text-[#054c5b]">Our Mission</h3>
            <p className="text-muted-foreground">To equip clinicians with reliable, interpretable AI tools that support early, confident cancer detection while safeguarding patient privacy.</p>
          </div>
        </section>
      </ScrollRevealSection>

      {/* FEATURES */}
      <ScrollRevealSection>
        <section className="py-16 sm:py-24 px-4 sm:px-6 lg:px-8 bg-background">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-4xl sm:text-5xl font-serif font-bold mb-4 text-balance text-[#054c5b]">Core Features</h2>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                Designed for clinical efficiency and accuracy
              </p>
            </div>

            <div className="">
              <Carousel>
                {/* Image Upload */}
                <Card className="border-border hover:shadow-sm transition-shadow scroll-reveal-child">
                  <CardHeader>
                    <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                      <Microscope className="w-6 h-6 text-primary" />
                    </div>
                    <CardTitle className="text-lg">Image Upload</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Drag-and-drop histopathological images with validation for quality and format
                    </p>
                  </CardContent>
                </Card>

                {/* AI Analysis */}
                <Card className="border-border hover:shadow-sm transition-shadow scroll-reveal-child">
                  <CardHeader>
                    <div className="w-12 h-12 rounded-lg bg-accent/10 flex items-center justify-center mb-4">
                      <BarChart3 className="w-6 h-6 text-accent" />
                    </div>
                    <CardTitle className="text-lg">AI Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Advanced neural networks analyze tissue regions with confidence scores and heatmaps
                    </p>
                  </CardContent>
                </Card>

                {/* Report Generation */}
                <Card className="border-border hover:shadow-sm transition-shadow scroll-reveal-child">
                  <CardHeader>
                    <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                      <FileText className="w-6 h-6 text-primary" />
                    </div>
                    <CardTitle className="text-lg">Report Generation</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Auto-generate clinical reports with interpretability explanations and export to PDF
                    </p>
                  </CardContent>
                </Card>

                {/* Role-Based Access */}
                <Card className="border-border hover:shadow-sm transition-shadow scroll-reveal-child">
                  <CardHeader>
                    <div className="w-12 h-12 rounded-lg bg-accent/10 flex items-center justify-center mb-4">
                      <Lock className="w-6 h-6 text-accent" />
                    </div>
                    <CardTitle className="text-lg">Role-Based Access</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Secure authentication with role-based permissions for Oncologists, Technicians, and Admins
                    </p>
                  </CardContent>
                </Card>
              </Carousel>
            </div>
          </div>
        </section>
      </ScrollRevealSection>

      {/* HOW IT WORKS */}
      <ScrollRevealSection>
        <section className="py-16 sm:py-24 px-4 sm:px-6 lg:px-8 bg-secondary/30">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl sm:text-5xl font-serif font-bold mb-12 text-center text-balance text-[#054c5b]">How It Works</h2>

            <div className="relative">
              {/* Vertical timeline line */}
              <div className="hidden sm:block absolute left-6 top-8 bottom-0 w-px bg-border" />

              <div className="space-y-8">
                {[
                  {
                    step: 1,
                    title: "Upload Image",
                    description: "Securely upload histopathological images with patient and case information",
                    icon: <Microscope className="w-5 h-5 text-primary" />,
                  },
                  {
                    step: 2,
                    title: "AI Processing",
                    description: "Our pre-trained model analyzes tissue patterns and generates diagnostic predictions",
                    icon: <BarChart3 className="w-5 h-5 text-accent" />,
                  },
                  {
                    step: 3,
                    title: "Review Analysis",
                    description: "Examine confidence scores, region-specific insights, and visual heatmaps",
                    icon: <ImageIcon className="w-5 h-5 text-primary" />,
                  },
                  {
                    step: 4,
                    title: "Generate Report",
                    description: "Export comprehensive clinical reports with explanations and PDF export",
                    icon: <FileText className="w-5 h-5 text-primary" />,
                  },
                ].map((item) => (
                  <div key={item.step} className="relative flex items-start gap-6 scroll-reveal-child">
                    <div className="flex-shrink-0 z-10">
                      <div className="w-12 h-12 rounded-full bg-card border border-border flex items-center justify-center shadow-sm">
                        {item.icon}
                      </div>
                    </div>
                    <div className="flex-1">
                      <div className="p-6 bg-card rounded-lg border border-border shadow-sm">
                        <h3 className="text-lg font-semibold mb-2">{item.title}</h3>
                        <p className="text-muted-foreground">{item.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </ScrollRevealSection>

      {/* FINAL HERO-STYLED CTA */}
      <ScrollRevealSection>
        <section className="py-12 px-6 bg-[#f6fbfc]">
          <div className="max-w-6xl mx-auto flex flex-col items-center justify-center text-center py-12">
            <h2 className="font-serif text-[52px] sm:text-[70px] font-bold leading-tight text-[#054c5b]">Experience the clarity of BioSight</h2>
            <div className="mt-8">
              <Link
                href="/auth"
                className="px-10 py-4 rounded-full text-white font-semibold text-lg shadow-md"
                style={{ backgroundColor: "#054c5b" }}
              >
                Try Now
              </Link>
            </div>
          </div>
        </section>
      </ScrollRevealSection>

      <Footer />
    </main>
  )
}
