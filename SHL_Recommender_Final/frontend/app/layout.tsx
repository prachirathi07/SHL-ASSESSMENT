import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'SHL Assessment Recommender',
  description: 'Find the perfect SHL assessments for your recruitment needs',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <div className="min-h-screen flex flex-col">
          {/* Animated background elements */}
          <div className="fixed top-0 left-0 w-full h-full z-0 overflow-hidden pointer-events-none">
            <div className="absolute top-20 left-20 w-64 h-64 rounded-full bg-primary/10 blur-3xl animate-pulse"></div>
            <div className="absolute bottom-20 right-20 w-80 h-80 rounded-full bg-secondary/10 blur-3xl animate-pulse" 
                 style={{ animationDelay: '2s', animationDuration: '6s' }}></div>
          </div>

          <header className="relative z-10 border-b border-border/50 bg-background/70 backdrop-blur-md">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center h-16">
                <div className="flex-shrink-0 flex items-center">
                  <span className="text-xl font-bold text-primary animate-glow">
                    SHL Assessment Recommender
                  </span>
                </div>
                <nav className="flex space-x-8 items-center">
                  <a href="/" className="text-text-secondary hover:text-primary transition-colors">
                    Home
                  </a>
                  <a href="/about" className="text-text-secondary hover:text-primary transition-colors">
                    About
                  </a>
                </nav>
              </div>
            </div>
          </header>

          <main className="flex-grow relative z-10">
            {children}
          </main>

          <footer className="relative z-10 border-t border-border/50 py-8 bg-background/70 backdrop-blur-md">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <p className="text-center text-text-muted">
                &copy; {new Date().getFullYear()} SHL Assessment Recommendation System
              </p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  )
} 