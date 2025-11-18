'use client'

import React, { useEffect, useRef, useState } from 'react'

interface ScrollRevealSectionProps {
  children: React.ReactNode
  className?: string
}

export default function ScrollRevealSection({
  children,
  className = '',
}: ScrollRevealSectionProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
          observer.unobserve(entry.target)
        }
      },
      { threshold: 0.1 }
    )

    if (ref.current) {
      observer.observe(ref.current)
    }

    return () => observer.disconnect()
  }, [])

  return (
    <div
      ref={ref}
      className={`${isVisible ? 'scroll-reveal-enter' : 'opacity-0'} ${className}`}
    >
      {children}
    </div>
  )
}
