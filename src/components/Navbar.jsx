import { Link, useLocation } from "react-router-dom"
import { motion } from "framer-motion"
import { Building2, MessageSquare, Briefcase, FileUp, BarChart3, Database, Target } from "lucide-react"

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 w-full border-b border-border bg-background/80 backdrop-blur-md">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 group">
          <div className="bg-primary p-2 rounded-lg">
            <Building2 className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-lg tracking-tight group-hover:text-primary transition-colors">
            MSME Viability
          </span>
        </Link>
        
        <div className="flex items-center">
          <a
            href="https://msme-viability-assessment.onrender.com/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-sm font-medium text-muted-foreground hover:text-primary transition-colors"
          >
            <Database className="w-4 h-4" />
            <span className="hidden sm:inline">API</span>
          </a>
        </div>
      </div>
    </nav>
  )
}
