import { Globe, Mail, Lightbulb, Code2, Phone, MessageSquareWarning } from "lucide-react"

export default function Footer() {
  return (
    <footer className="w-full border-t border-slate-200 bg-white mt-12 py-10">
      <div className="container mx-auto px-4 grid grid-cols-1 md:grid-cols-3 gap-8">
        
        <div className="space-y-3">
          <h4 className="font-bold text-slate-800 flex items-center gap-2">
            <Code2 className="w-5 h-5 text-primary" /> About the Project
          </h4>
          <p className="text-sm text-slate-500 leading-relaxed">
            The MSME Viability Assessment is an advanced AI-driven platform designed to evaluate loan readiness using state-of-the-art machine learning (XGBoost) and Large Language Models. It provides transparent, real-time financial risk diagnostics to bridge the credit gap.
          </p>
        </div>

        <div className="space-y-3">
          <h4 className="font-bold text-slate-800 flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-primary" /> For Whom?
          </h4>
          <p className="text-sm text-slate-500 leading-relaxed">
            Designed for <strong>MSME Founders</strong> seeking clear, actionable insights into their loan eligibility, and <strong>Financial Institutions</strong> looking to perform deep scenario testing and streamline their credit evaluation processes.
          </p>
        </div>

        <div className="space-y-3">
          <h4 className="font-bold text-slate-800">Developer Info</h4>
          <ul className="space-y-2 text-sm text-slate-500">
            <li><strong>Author:</strong> Pashin (AIML Engineer)</li>
            <li>
              <a href="https://github.com/PashinP/msme-viability-assessment" target="_blank" rel="noreferrer" className="flex items-center gap-2 hover:text-primary transition-colors">
                <Globe className="w-4 h-4" /> View Architecture on GitHub
              </a>
            </li>
            <li>
              <a href="mailto:pashinpruthiworking@gmail.com" className="flex items-center gap-2 hover:text-primary transition-colors">
                <Mail className="w-4 h-4" /> pashinpruthiworking@gmail.com
              </a>
            </li>
            <li className="flex items-center gap-2">
              <Phone className="w-4 h-4" /> +91 6395867970
            </li>
            <li className="pt-2">
              <a href="mailto:pashinpruthiworking@gmail.com?subject=MSME Dashboard Feedback" className="inline-flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors border border-slate-200">
                <MessageSquareWarning className="w-3.5 h-3.5" /> Report Bug / Feedback
              </a>
            </li>
          </ul>
        </div>
        
      </div>
      <div className="container mx-auto px-4 mt-8 pt-6 border-t border-slate-100 text-center text-xs text-slate-400">
        © {new Date().getFullYear()} MSME Viability Assessment. Built for demonstration purposes.
      </div>
    </footer>
  )
}
