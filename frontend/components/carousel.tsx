"use client"

import React from 'react'
import useEmblaCarousel from 'embla-carousel-react'

export default function Carousel({ children }: { children: React.ReactNode[] }) {
  const [emblaRef, emblaApi] = useEmblaCarousel({ align: 'center', loop: false })
  const [selected, setSelected] = React.useState(0)

  React.useEffect(() => {
    if (!emblaApi) return
    const onSelect = () => setSelected(emblaApi.selectedScrollSnap())
    emblaApi.on('select', onSelect)
    onSelect()
    return () => emblaApi.off('select', onSelect)
  }, [emblaApi])

  const getSlideClasses = (idx: number) => {
    const isSelected = idx === selected
    const baseClasses = 'transition-all duration-500 ease-out flex-shrink-0'
    
    if (isSelected) {
      return `${baseClasses} min-w-[80%] md:min-w-[45%] lg:min-w-[22%] scale-100 opacity-100`
    } else {
      return `${baseClasses} min-w-[80%] md:min-w-[45%] lg:min-w-[22%] scale-90 opacity-40 blur-[5px]`
    }
  }

  return (
    <div className="relative px-4 md:px-0">
      {/* Carousel viewport */}
      <div className="overflow-hidden rounded-xl" ref={emblaRef}>
        <div className="flex gap-6 py-4">
          {React.Children.map(children, (child, idx) => (
            <div
              className={getSlideClasses(idx)}
              key={idx}
            >
              <div
                className={`transition-all duration-500 ease-out h-full ${
                  idx === selected
                    ? 'shadow-lg shadow-black/10 rounded-xl'
                    : 'shadow-sm shadow-black/5 rounded-lg'
                }`}
              >
                {child}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="absolute -translate-y-1/2 top-1/2 left-0 md:left-2 z-10">
        <button
          aria-label="Previous"
          className="bg-white hover:bg-gray-50 text-gray-700 p-2.5 rounded-full shadow-md hover:shadow-lg transition-all duration-300 hover:scale-110"
          onClick={() => emblaApi && emblaApi.scrollPrev()}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
      </div>
      <div className="absolute -translate-y-1/2 top-1/2 right-0 md:right-2 z-10">
        <button
          aria-label="Next"
          className="bg-white hover:bg-gray-50 text-gray-700 p-2.5 rounded-full shadow-md hover:shadow-lg transition-all duration-300 hover:scale-110"
          onClick={() => emblaApi && emblaApi.scrollNext()}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>

      {/* Indicator dots */}
      <div className="flex items-center justify-center gap-2 mt-8">
        {Array.from({ length: React.Children.count(children) }).map((_, i) => (
          <button
            key={i}
            className={`transition-all duration-400 ease-out rounded-full ${
              i === selected
                ? 'w-8 h-2 bg-foreground/80'
                : 'w-2 h-2 bg-gray-300 hover:bg-gray-400'
            }`}
            aria-label={`Go to slide ${i + 1}`}
            onClick={() => emblaApi && emblaApi.scrollTo(i)}
          />
        ))}
      </div>
    </div>
  )
}
