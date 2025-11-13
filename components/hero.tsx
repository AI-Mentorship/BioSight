export default function Hero() {
  const handleTryNow = () => {
    // Scroll to upload section
    const uploadSection = document.getElementById('upload');
    uploadSection?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleLearnMore = () => {
    // Scroll to models section
    const modelsSection = document.getElementById('models');
    modelsSection?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section
      id="home"
      className="relative py-20 lg:py-32 bg-gradient-to-br from-primary/5 via-background to-background"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div className="space-y-6">
            <div className="inline-block px-4 py-2 bg-primary/10 rounded-full border border-primary/20">
              <span className="text-sm font-semibold text-primary">Advanced AI-Powered Detection</span>
            </div>
            <h1 className="text-5xl lg:text-6xl font-bold text-foreground leading-tight">
              Early Cancer Detection
              <span className="block text-primary">Powered by AI</span>
            </h1>
            <p className="text-lg text-muted-foreground leading-relaxed">
              Revolutionizing healthcare with advanced machine learning models that detect multiple cancer types with
              exceptional accuracy. Our platform helps healthcare professionals make informed decisions faster.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 pt-6">
              <button 
                onClick={handleTryNow}
                className="px-8 py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:opacity-90 transition-opacity"
              >
                Try Now
              </button>
              <button 
                onClick={handleLearnMore}
                className="px-8 py-3 border border-primary text-primary rounded-lg font-semibold hover:bg-primary/5 transition-colors"
              >
                Learn More
              </button>
            </div>
          </div>

          {/* Right Visual */}
          <div className="relative h-96 lg:h-full min-h-96">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-accent/20 rounded-2xl blur-3xl"></div>
            <div className="relative h-full flex items-center justify-center">
              <div className="w-64 h-64 bg-primary/10 rounded-3xl border border-primary/20 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-5xl font-bold text-primary mb-2">98%</div>
                  <p className="text-sm text-muted-foreground">Detection Accuracy</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}