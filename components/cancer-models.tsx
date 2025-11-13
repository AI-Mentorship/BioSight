"use client"

import { useState } from "react"
import CancerCard from "./cancer-card"

const cancerTypes = [
  {
    id: 1,
    name: "Lymphoma",
    description: "Detection of lymphoid tissue cancers affecting blood cells and immune system",
    icon: "🔬",
    accuracy: "96%",
    color: "from-blue-500/20 to-blue-600/20",
  },
  {
    id: 2,
    name: "Breast Cancer",
    description: "Advanced screening for early-stage breast cancer detection",
    icon: "💙",
    accuracy: "98%",
    color: "from-pink-500/20 to-pink-600/20",
  },
  {
    id: 3,
    name: "Lung Cancer",
    description: "Rapid identification of lung tissue abnormalities",
    icon: "💨",
    accuracy: "97%",
    color: "from-green-500/20 to-green-600/20",
  },
  {
    id: 4,
    name: "Oral Cancer",
    description: "Detection of cancerous lesions in oral cavity",
    icon: "😊",
    accuracy: "95%",
    color: "from-orange-500/20 to-orange-600/20",
  },
  {
    id: 5,
    name: "Cervical Cancer",
    description: "Early detection through cellular analysis",
    icon: "🔍",
    accuracy: "96%",
    color: "from-purple-500/20 to-purple-600/20",
  },
  {
    id: 6,
    name: "Colon Cancer",
    description: "Identification of colorectal tissue abnormalities",
    icon: "🫘",
    accuracy: "97%",
    color: "from-amber-500/20 to-amber-600/20",
  },
]

export default function CancerModels() {
  const [selectedTab, setSelectedTab] = useState(0)

  return (
    <section id="models" className="py-20 lg:py-32 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-4">Our Cancer Detection Models</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Six advanced AI models trained on extensive medical datasets for accurate cancer detection
          </p>
        </div>

        {/* Tab Navigation for Mobile */}
        <div className="md:hidden mb-8 overflow-x-auto">
          <div className="flex gap-2 pb-2">
            {cancerTypes.map((cancer, index) => (
              <button
                key={cancer.id}
                onClick={() => setSelectedTab(index)}
                className={`px-4 py-2 rounded-lg whitespace-nowrap transition-colors ${
                  selectedTab === index
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground hover:bg-border"
                }`}
              >
                {cancer.name}
              </button>
            ))}
          </div>
        </div>

        {/* Mobile Selected Card View */}
        <div className="md:hidden mb-8">
          <CancerCard cancer={cancerTypes[selectedTab]} />
        </div>

        {/* Desktop Grid */}
        <div className="hidden md:grid grid-cols-1 lg:grid-cols-3 gap-6">
          {cancerTypes.map((cancer) => (
            <CancerCard key={cancer.id} cancer={cancer} />
          ))}
        </div>
      </div>
    </section>
  )
}
