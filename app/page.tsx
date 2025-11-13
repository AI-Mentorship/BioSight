"use client";

import { ThemeProvider } from "../components/theme-provider";
import Navbar from "../components/navbar";
import Hero from "../components/hero";
import CancerModels from "../components/cancer-models";
import ImageUpload from "../components/image-upload";
import Footer from "../components/footer";

export default function Home() {
  return (
    <ThemeProvider>
      <main className="min-h-screen bg-background text-foreground">
        <Navbar />
        <Hero />
        <CancerModels />
        <ImageUpload />
        <Footer />
      </main>
    </ThemeProvider>
  );
}
