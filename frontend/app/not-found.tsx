import Link from "next/link"
import { Header } from "@/components/header"
import { Footer } from "@/components/footer"
import { Button } from "@/components/ui/button"

export default function NotFound() {
  return (
    <>
      <Header />
      <main className="flex-1 flex items-center justify-center px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center">
          <h1 className="text-6xl font-bold text-foreground mb-4">404</h1>
          <p className="text-xl text-muted-foreground mb-8">Page not found</p>
          <Button asChild>
            <Link href="/">Return Home</Link>
          </Button>
        </div>
      </main>
      <Footer />
    </>
  )
}
