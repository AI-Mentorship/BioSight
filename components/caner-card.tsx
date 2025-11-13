"use client"

interface CancerCardProps {
  cancer: {
    id: number
    name: string
    description: string
    icon: string
    accuracy: string
    color: string
  }
}

export default function CancerCard({ cancer }: CancerCardProps) {
  const handleTryModel = () => {
    // Scroll to upload section and optionally pre-select this cancer type
    const uploadSection = document.getElementById('upload');
    uploadSection?.scrollIntoView({ behavior: 'smooth' });
    
    // You could also set the selected model in localStorage or context
    // For now, we'll just scroll to the upload section
    console.log(`Selected model: ${cancer.name}`);
  }

  return (
    <div
      className={`group p-6 rounded-xl border border-border bg-card hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:shadow-primary/10 cursor-pointer transform hover:scale-105`}
    >
      {/* Icon */}
      <div className="text-4xl mb-4">{cancer.icon}</div>

      {/* Title */}
      <h3 className="text-2xl font-bold text-foreground mb-2">{cancer.name}</h3>

      {/* Accuracy Badge */}
      <div className="inline-block mb-4 px-3 py-1 bg-primary/10 rounded-full">
        <span className="text-sm font-semibold text-primary">{cancer.accuracy} Accuracy</span>
      </div>

      {/* Description */}
      <p className="text-muted-foreground mb-6">{cancer.description}</p>

      {/* Try Model Button */}
      <button 
        onClick={handleTryModel}
        className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg font-semibold hover:opacity-90 transition-opacity"
      >
        Try Model →
      </button>
    </div>
  )
}
