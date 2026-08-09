import { useState, useEffect, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import axios from "axios"
import { Send, Upload, Settings2, Paperclip, Activity, FileText, Download, ChevronRight, BarChart3, ShieldCheck, Zap, Database, Bot, User, Users, X, ChevronUp, ChevronDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import AssessmentRadarChart from "@/components/RadarChart"
import ShapWaterfall from "@/components/ShapWaterfall"
import ReadinessPanel from "@/components/ReadinessPanel"

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"
const API_KEY = import.meta.env.VITE_API_KEY || "msme-dev-key-2024"

// ─── Simple Markdown → HTML for chat messages ────────────
function renderMarkdown(text) {
  if (!text) return ""
  let html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')  // escape HTML
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')                     // **bold**
    .replace(/\*(.+?)\*/g, '<em>$1</em>')                                 // *italic*
    .replace(/`(.+?)`/g, '<code style="background:#f1f5f9;padding:1px 4px;border-radius:3px;font-size:0.85em">$1</code>') // `code`
    .replace(/\n/g, '<br/>')                                              // newlines
  return html
}

const CORE_KEYS = [
  "Term", "NoEmp", "NewExist", "CreateJob", "RetainedJob", 
  "DisbursementGross", "UrbanRural", "RevLineCr", "LowDoc",
  "SBA_Appv", "GrAppv"
]

// ─── Hero Showcase Component (Empty State) ────────────────
const HeroShowcase = () => (
  <div className="flex-1 rounded-2xl flex flex-col h-full bg-white border border-slate-200 overflow-hidden shadow-sm">
    <div className="bg-slate-900 p-8 text-white">
      <div className="flex items-center gap-2 text-blue-400 mb-3 text-sm font-semibold tracking-wider uppercase">
        <ShieldCheck className="w-4 h-4" /> Enterprise-Grade AI
      </div>
      <h2 className="text-3xl font-bold mb-2">MSME Viability Assessment Engine</h2>
      <p className="text-slate-400 max-w-xl text-sm leading-relaxed">
        A multi-model architecture combining NLP data extraction, gradient boosted trees, and SHAP explainability to evaluate business loan readiness with institutional precision.
      </p>
    </div>
    
    <div className="p-8 flex-1 bg-slate-50 overflow-y-auto">
      <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">Core Architecture</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="w-10 h-10 bg-blue-50 text-blue-600 rounded-lg flex items-center justify-center mb-3">
            <FileText className="w-5 h-5" />
          </div>
          <h4 className="font-bold text-slate-800 text-sm mb-1">1. Intelligent Ingestion</h4>
          <p className="text-xs text-slate-500 leading-relaxed">Extracts 25+ parameters from conversational text or unstructured documents using Llama-3 NLP processing.</p>
        </div>
        
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm relative">
          <div className="absolute -left-3 top-1/2 -translate-y-1/2 w-6 h-6 bg-slate-50 border border-slate-200 rounded-full flex items-center justify-center text-slate-400 z-10 hidden md:flex">
            <ChevronRight className="w-3 h-3" />
          </div>
          <div className="w-10 h-10 bg-indigo-50 text-indigo-600 rounded-lg flex items-center justify-center mb-3">
            <Database className="w-5 h-5" />
          </div>
          <h4 className="font-bold text-slate-800 text-sm mb-1">2. Multi-Model Scoring</h4>
          <p className="text-xs text-slate-500 leading-relaxed">Ensembles XGBoost and LightGBM models trained on 897,167 historical SBA loans to predict default probability.</p>
        </div>
        
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm relative">
           <div className="absolute -left-3 top-1/2 -translate-y-1/2 w-6 h-6 bg-slate-50 border border-slate-200 rounded-full flex items-center justify-center text-slate-400 z-10 hidden md:flex">
            <ChevronRight className="w-3 h-3" />
          </div>
          <div className="w-10 h-10 bg-emerald-50 text-emerald-600 rounded-lg flex items-center justify-center mb-3">
            <Zap className="w-5 h-5" />
          </div>
          <h4 className="font-bold text-slate-800 text-sm mb-1">3. Actionable Insights</h4>
          <p className="text-xs text-slate-500 leading-relaxed">Generates specific prescriptions and government scheme matches utilizing SHAP explainability analysis.</p>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-100 rounded-xl p-5 flex items-start gap-4">
         <div className="mt-1">
           <BarChart3 className="w-5 h-5 text-blue-500" />
         </div>
         <div>
           <h4 className="font-bold text-blue-900 text-sm mb-1">Ready to run an assessment?</h4>
           <p className="text-xs text-blue-700 max-w-lg leading-relaxed">
             Describe your business in the chat (revenue, employees, requested loan amount, etc.), or click the ⚙️ icon to manually configure the parameters.
           </p>
         </div>
      </div>
    </div>
  </div>
)

// ─── Inline Slider for the Tweak Panel ───────────────────
function TweakSlider({ label, value, min, max, step, unit, onChange }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-500">{label}</span>
        <span className="font-semibold text-slate-700">{unit === "₹" ? `₹${Number(value).toLocaleString('en-IN')}` : value}{unit === "mo" ? " months" : ""}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-slate-200 rounded-full appearance-none cursor-pointer accent-blue-500" />
    </div>
  )
}

function TweakSelect({ label, value, options, onChange }) {
  return (
    <div>
      <label className="text-xs text-slate-500 block mb-1">{label}</label>
      <select value={value} onChange={e => onChange(e.target.value)}
        className="w-full text-xs bg-white border border-slate-200 rounded-lg px-2 py-1.5 focus:outline-none focus:border-blue-400">
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  )
}


export default function Dashboard() {
  // -- State: Core ML Features --
  const [features, setFeatures] = useState({
    Term: 84, NoEmp: 5, NewExist: 1, CreateJob: 2, RetainedJob: 5,
    DisbursementGross: 100000, UrbanRural: 1, RevLineCr: 0, LowDoc: 0,
    SBA_Appv: 75000, GrAppv: 100000
  })

  // -- State: Business Context (new expanded fields) --
  const [businessContext, setBusinessContext] = useState({
    monthly_revenue: 200000,
    monthly_expenses: 150000,
    existing_debt_emi: 0,
    years_in_operation: 3,
    industry_sector: "Retail",
    has_gst: true,
    has_udyam: false,
    collateral_value: 0,
    tax_filing_years: 2,
    loan_purpose: "Working Capital",
  })

  // -- State: Chat --
  const [messages, setMessages] = useState([
    { role: "assistant", content: "👋 नमस्ते! I'm your Loan Readiness Coach. Tell me about your business — what do you do, where are you located, and what kind of loan are you looking for?\n\nYou can also attach business documents using the 📎 button, or tweak numbers manually using the ⚙️ button." }
  ])
  const [chatInput, setChatInput] = useState("")
  const [isChatLoading, setIsChatLoading] = useState(false)
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)

  // -- State: Results --
  const [results, setResults] = useState(null)
  const [assessmentData, setAssessmentData] = useState(null)
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false)
  const [pdfReadyUrl, setPdfReadyUrl] = useState(null)
  const [isEvaluating, setIsEvaluating] = useState(false)

  // -- State: UI --
  const [showTweakPanel, setShowTweakPanel] = useState(false)

  // ─── Fetch Assessment ─────────────────────────────────
  const fetchAssessment = async (featureSet, contextData = {}) => {
    setIsEvaluating(true)
    setPdfReadyUrl(null)
    try {
      const headers = { "X-API-Key": API_KEY }

      const coreFeatures = {}
      const extractedContext = { ...contextData }
      for (const [k, v] of Object.entries(featureSet)) {
        if (CORE_KEYS.includes(k)) coreFeatures[k] = v
        else extractedContext[k] = v
      }

      const assessRes = await axios.post(`${API_URL}/assess`, {
        features: coreFeatures,
        context: extractedContext
      }, { headers })

      setAssessmentData(assessRes.data)

      const [expl, sim] = await Promise.all([
        axios.post(`${API_URL}/explain`, coreFeatures, { headers }),
        axios.post(`${API_URL}/similar`, coreFeatures, { headers }).catch(() => ({ data: null })),
      ])

      setResults({
        prediction: assessRes.data.prediction,
        explanation: expl.data,
        similar: sim.data,
      })
    } catch (err) {
      console.error("Assessment error:", err)
    } finally {
      setIsEvaluating(false)
    }
  }

  // ─── Chat Submission ──────────────────────────────────
  const handleChatSubmit = async (e) => {
    e?.preventDefault()
    if (!chatInput.trim() || isChatLoading) return

    const userMessage = { role: "user", content: chatInput.trim() }
    setMessages(prev => [...prev, userMessage])
    setChatInput("")
    setIsChatLoading(true)

    try {
      const historyForApi = messages.map(m => ({
        role: m.role === "assistant" ? "model" : "user",
        content: m.content
      })).concat([{ role: "user", content: userMessage.content }])

      const response = await axios.post(`${API_URL}/chat`, { messages: historyForApi }, {
        headers: { "X-API-Key": API_KEY }
      })

      const { response: aiResponse, extraction_complete, features_extracted } = response.data

      const displayResponse = aiResponse.replace(/```json[\s\S]*?```/g, '').trim()
      setMessages(prev => [...prev, { role: "assistant", content: displayResponse || aiResponse }])

      if (extraction_complete && features_extracted) {
        const coreFeatures = {}
        const ctx = {}
        for (const [k, v] of Object.entries(features_extracted)) {
          if (CORE_KEYS.includes(k)) coreFeatures[k] = v
          else ctx[k] = v
        }
        setFeatures(prev => ({ ...prev, ...coreFeatures }))
        setBusinessContext(prev => ({ ...prev, ...ctx }))
        fetchAssessment(coreFeatures, ctx)
      }
    } catch (error) {
      console.error("Chat error:", error)
      setMessages(prev => [...prev, { role: "assistant", content: "⚠️ Connection error. Please check that the backend server is running." }])
    } finally {
      setIsChatLoading(false)
    }
  }

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }) }, [messages])

  // ─── File Attachment ──────────────────────────────────
  const handleFileAttach = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (file.type === "text/plain" || file.type === "text/csv" || file.name.endsWith('.txt') || file.name.endsWith('.csv')) {
      const text = await file.text()
      const truncated = text.substring(0, 3000)
      setChatInput(prev => prev + `\n\n📄 [${file.name}]:\n${truncated}`)
      setMessages(prev => [...prev, { role: "system", content: `📎 File attached: ${file.name}` }])
    } else {
      setMessages(prev => [...prev, { role: "system", content: `📎 ${file.name} attached. For best results, paste the key numbers from your document directly in the chat.` }])
    }
    e.target.value = ""
  }

  // ─── Manual Tweak: Run Assessment ─────────────────────
  const handleTweakRun = () => {
    // Merge features with context and run
    const mergedFeatures = { ...features, DisbursementGross: features.GrAppv || features.DisbursementGross }
    fetchAssessment(mergedFeatures, businessContext)
    setShowTweakPanel(false)
    setMessages(prev => [...prev, { role: "system", content: "⚙️ Manual assessment triggered with updated parameters." }])
  }

  // ─── PDF Download ─────────────────────────────────────
  const handleDownloadPDF = async () => {
    if (!results || isEvaluating) return
    setIsGeneratingPdf(true)
    try {
      const coreFeatures = {}
      const extractedContext = { ...businessContext }
      for (const [k, v] of Object.entries(features)) {
        if (CORE_KEYS.includes(k)) coreFeatures[k] = v
        else extractedContext[k] = v
      }

      const response = await axios.post(`${API_URL}/report`, {
        features: coreFeatures,
        context: extractedContext
      }, {
        headers: { "X-API-Key": API_KEY },
        responseType: 'blob'
      })
      
      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      setPdfReadyUrl(url);
      
    } catch (err) {
      console.error("Failed to download PDF", err)
      const errorDetail = err.response ? `Server Error: ${err.response.status}` : err.message;
      alert(`PDF generation failed: ${errorDetail}. Check console for details.`)
    } finally {
      setIsGeneratingPdf(false)
    }
  }

  return (
    <div className="container mx-auto py-6 px-4">
      
      {/* Header */}
      <div className="mb-6 flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Loan Readiness Coach</h1>
          <p className="text-sm text-slate-500">AI-powered business loan assessment for MSMEs</p>
        </div>
        {assessmentData && (
          pdfReadyUrl ? (
            <a 
              href={pdfReadyUrl} 
              download={`MSME_Assessment_${new Date().getTime()}.pdf`}
              className="inline-flex items-center justify-center whitespace-nowrap rounded-md gap-2 border border-green-200 text-green-700 bg-green-50 hover:bg-green-100 px-4 py-2 text-sm font-medium transition-colors"
            >
              <Download className="w-4 h-4" />
              Download Ready!
            </a>
          ) : (
            <Button 
              onClick={handleDownloadPDF} 
              variant="outline" 
              className="gap-2 border-blue-200 text-blue-600 hover:bg-blue-50 text-sm"
              disabled={isGeneratingPdf}
            >
              {isGeneratingPdf ? (
                <>
                  <div className="w-4 h-4 rounded-full border-2 border-blue-600 border-t-transparent animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  Export PDF
                </>
              )}
            </Button>
          )
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6" style={{ height: "calc(100vh - 10rem)" }}>
        
        {/* ════════ LEFT: Chat Panel ════════ */}
        <div className="lg:col-span-5 flex flex-col glass-panel rounded-2xl overflow-hidden relative">
          
          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-hide">
            <AnimatePresence>
              {messages.map((msg, idx) => {
                if (msg.role === "system") {
                  return (
                    <motion.div key={idx} initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                      className="text-center text-xs text-slate-400 py-1">
                      {msg.content}
                    </motion.div>
                  )
                }
                return (
                  <motion.div key={idx} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                    className={`flex items-start gap-2.5 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${msg.role === "user" ? "bg-blue-100" : "bg-slate-100"}`}>
                      {msg.role === "user" ? <User className="w-3.5 h-3.5 text-blue-600" /> : <Bot className="w-3.5 h-3.5 text-slate-500" />}
                    </div>
                    <div className={`px-3.5 py-2.5 rounded-2xl max-w-[85%] text-sm leading-relaxed ${
                      msg.role === "user" 
                        ? "bg-blue-500 text-white rounded-tr-sm" 
                        : "bg-slate-50 border border-slate-100 rounded-tl-sm text-slate-700"
                    }`}
                      dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                    />
                  </motion.div>
                )
              })}
              {isChatLoading && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-start gap-2.5">
                  <div className="w-7 h-7 rounded-full bg-slate-100 flex items-center justify-center flex-shrink-0">
                    <Bot className="w-3.5 h-3.5 text-slate-500" />
                  </div>
                  <div className="bg-slate-50 border border-slate-100 px-4 py-3 rounded-2xl rounded-tl-sm flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>

          {/* ── Slide-up Tweak Panel ── */}
          <AnimatePresence>
            {showTweakPanel && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="overflow-hidden border-t border-slate-200 bg-slate-50/80 backdrop-blur-sm"
              >
                <div className="p-4 space-y-4 max-h-[50vh] overflow-y-auto">
                  <div className="flex justify-between items-center">
                    <h4 className="text-xs font-bold text-slate-600 uppercase tracking-wider">Quick Tweak Parameters</h4>
                    <button onClick={() => setShowTweakPanel(false)} className="text-slate-400 hover:text-slate-600">
                      <X className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Business Context */}
                  <div className="space-y-3">
                    <div className="text-[10px] font-semibold text-blue-500 uppercase tracking-wider">Business Info</div>
                    <TweakSlider label="Monthly Revenue" value={businessContext.monthly_revenue} min={0} max={5000000} step={10000} unit="₹"
                      onChange={v => setBusinessContext(p => ({ ...p, monthly_revenue: v }))} />
                    <TweakSlider label="Monthly Expenses" value={businessContext.monthly_expenses} min={0} max={5000000} step={10000} unit="₹"
                      onChange={v => setBusinessContext(p => ({ ...p, monthly_expenses: v }))} />
                    <TweakSlider label="Existing EMI" value={businessContext.existing_debt_emi} min={0} max={500000} step={1000} unit="₹"
                      onChange={v => setBusinessContext(p => ({ ...p, existing_debt_emi: v }))} />
                    <TweakSlider label="Years in Operation" value={businessContext.years_in_operation} min={0} max={30} step={1} unit=""
                      onChange={v => setBusinessContext(p => ({ ...p, years_in_operation: v }))} />
                    <TweakSlider label="Collateral Value" value={businessContext.collateral_value} min={0} max={50000000} step={100000} unit="₹"
                      onChange={v => setBusinessContext(p => ({ ...p, collateral_value: v }))} />
                    <TweakSlider label="ITR Filed (Years)" value={businessContext.tax_filing_years} min={0} max={10} step={1} unit=""
                      onChange={v => setBusinessContext(p => ({ ...p, tax_filing_years: v }))} />
                    <div className="grid grid-cols-2 gap-3">
                      <TweakSelect label="Industry" value={businessContext.industry_sector} onChange={v => setBusinessContext(p => ({ ...p, industry_sector: v }))}
                        options={[
                          { value: "Retail", label: "Retail" }, { value: "Manufacturing", label: "Manufacturing" },
                          { value: "Food & Beverage", label: "Food & Beverage" }, { value: "Services", label: "Services" },
                          { value: "Textile & Garments", label: "Textile" }, { value: "Agriculture", label: "Agriculture" },
                          { value: "Technology", label: "Technology" }, { value: "Healthcare", label: "Healthcare" },
                          { value: "Construction", label: "Construction" }, { value: "Other", label: "Other" },
                        ]} />
                      <TweakSelect label="Loan Purpose" value={businessContext.loan_purpose} onChange={v => setBusinessContext(p => ({ ...p, loan_purpose: v }))}
                        options={[
                          { value: "Working Capital", label: "Working Capital" }, { value: "Equipment Purchase", label: "Equipment" },
                          { value: "Expansion", label: "Expansion" }, { value: "New Venture", label: "New Venture" },
                          { value: "Inventory", label: "Inventory" }, { value: "Renovation", label: "Renovation" },
                        ]} />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <TweakSelect label="GST Registered" value={businessContext.has_gst ? "yes" : "no"}
                        onChange={v => setBusinessContext(p => ({ ...p, has_gst: v === "yes" }))}
                        options={[{ value: "yes", label: "Yes" }, { value: "no", label: "No" }]} />
                      <TweakSelect label="Udyam Registered" value={businessContext.has_udyam ? "yes" : "no"}
                        onChange={v => setBusinessContext(p => ({ ...p, has_udyam: v === "yes" }))}
                        options={[{ value: "yes", label: "Yes" }, { value: "no", label: "No" }]} />
                    </div>
                  </div>

                  {/* Loan Parameters */}
                  <div className="space-y-3 pt-2 border-t border-slate-200">
                    <div className="text-[10px] font-semibold text-blue-500 uppercase tracking-wider">Loan Parameters</div>
                    <TweakSlider label="Loan Amount (₹)" value={(features.GrAppv || 100000) * 83} min={50000} max={200000000} step={50000} unit="₹"
                      onChange={v => {
                        const usd = Math.round(v / 83)
                        setFeatures(p => ({ ...p, GrAppv: usd, DisbursementGross: usd, SBA_Appv: Math.round(usd * 0.75) }))
                      }} />
                    <TweakSlider label="Loan Term" value={features.Term || 84} min={12} max={240} step={12} unit="mo"
                      onChange={v => setFeatures(p => ({ ...p, Term: v }))} />
                    <div className="grid grid-cols-2 gap-3">
                      <TweakSlider label="Employees" value={features.NoEmp || 0} min={0} max={50} step={1} unit=""
                        onChange={v => setFeatures(p => ({ ...p, NoEmp: v }))} />
                      <TweakSlider label="New Hires Planned" value={features.CreateJob || 0} min={0} max={20} step={1} unit=""
                        onChange={v => setFeatures(p => ({ ...p, CreateJob: v }))} />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <TweakSelect label="Business Type" value={features.NewExist || 1}
                        onChange={v => setFeatures(p => ({ ...p, NewExist: Number(v) }))}
                        options={[{ value: 1, label: "Existing" }, { value: 2, label: "New Startup" }]} />
                      <TweakSelect label="Location" value={features.UrbanRural || 1}
                        onChange={v => setFeatures(p => ({ ...p, UrbanRural: Number(v) }))}
                        options={[{ value: 1, label: "Urban" }, { value: 2, label: "Rural" }]} />
                    </div>
                  </div>

                  {/* Run Button */}
                  <Button onClick={handleTweakRun} disabled={isEvaluating}
                    className="w-full py-3 text-sm font-semibold bg-blue-500 hover:bg-blue-600 text-white rounded-xl">
                    {isEvaluating ? (
                      <span className="flex items-center gap-2">
                        <span className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
                        Analyzing...
                      </span>
                    ) : (
                      <span className="flex items-center gap-2">
                        <Activity className="w-4 h-4" /> Run Assessment
                      </span>
                    )}
                  </Button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ── Chat Input Bar ── */}
          <div className="p-3 bg-white border-t border-slate-100">
            <form onSubmit={handleChatSubmit} className="flex items-center gap-2">
              {/* Attachment button */}
              <input type="file" ref={fileInputRef} onChange={handleFileAttach} className="hidden" accept=".txt,.csv,.pdf,.json" />
              <button type="button" onClick={() => fileInputRef.current?.click()}
                className="w-8 h-8 rounded-full flex items-center justify-center text-slate-400 hover:text-blue-500 hover:bg-blue-50 transition-colors"
                title="Attach business document">
                <Paperclip className="w-4 h-4" />
              </button>

              {/* Tweak button */}
              <button type="button" onClick={() => setShowTweakPanel(!showTweakPanel)}
                className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${showTweakPanel ? "text-blue-500 bg-blue-50" : "text-slate-400 hover:text-blue-500 hover:bg-blue-50"}`}
                title="Tweak parameters manually">
                <Settings2 className="w-4 h-4" />
              </button>

              {/* Text input */}
              <textarea 
                value={chatInput} 
                onChange={(e) => {
                  setChatInput(e.target.value)
                  e.target.style.height = 'auto'
                  e.target.style.height = (e.target.scrollHeight) + 'px'
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleChatSubmit(e)
                  }
                }}
                rows={1}
                placeholder="Tell me about your business..."
                className="flex-1 bg-slate-50 border border-slate-200 rounded-2xl pl-4 pr-4 py-2.5 text-sm focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 transition-all resize-none max-h-32"
                style={{ minHeight: '44px' }}
                disabled={isChatLoading} 
              />

              {/* Send button */}
              <Button type="submit" size="icon" className="w-8 h-8 rounded-full bg-blue-500 hover:bg-blue-600 flex-shrink-0"
                disabled={!chatInput.trim() || isChatLoading}>
                <Send className="w-3.5 h-3.5" />
              </Button>
            </form>
          </div>
        </div>

        {/* ════════ RIGHT: Results Panel ════════ */}
        <div className="lg:col-span-7 flex flex-col overflow-y-auto pr-1 space-y-5">
          {!assessmentData && !results ? (
             isEvaluating ? (
               <div className="flex-1 glass-panel rounded-2xl flex flex-col items-center justify-center p-10 text-center h-full">
                  <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4" />
                  <h3 className="text-lg font-bold text-slate-800">Analyzing Business Profile...</h3>
                  <p className="text-sm text-slate-500 mt-2 max-w-md">Running deep analysis across 6 dimensions of loan readiness.</p>
               </div>
             ) : (
               <HeroShowcase />
             )
          ) : (
             <div className="space-y-5">
                {/* Readiness Panel — the star */}
                {assessmentData && <ReadinessPanel assessmentData={assessmentData} />}

                {/* SHAP + Radar row */}
                {results?.explanation && (
                  <div className="flex flex-col lg:flex-row gap-5">
                    <div className="w-full lg:w-1/2 glass-panel p-5 rounded-2xl">
                      <h3 className="text-xs font-semibold mb-2 text-slate-400 uppercase tracking-wider">ML Feature Impact</h3>
                      <div className="h-52">
                        <ShapWaterfall explanation={results.explanation} />
                      </div>
                    </div>
                    <div className="w-full lg:w-1/2 glass-panel p-5 rounded-2xl flex flex-col items-center">
                      <h3 className="text-xs font-semibold mb-2 text-slate-400 uppercase tracking-wider w-full text-center">Business Profile Radar</h3>
                      <div className="flex-1 w-full h-52">
                        <AssessmentRadarChart features={features} />
                      </div>
                    </div>
                  </div>
                )}

                {/* Similar Businesses */}
                {results?.similar?.similar_loans?.length > 0 && (
                  <div className="glass-panel p-5 rounded-2xl">
                    <h3 className="text-xs font-semibold mb-3 text-slate-400 flex items-center gap-2 uppercase tracking-wider">
                      <Users className="w-3.5 h-3.5" /> Similar Historical Loans
                    </h3>
                    <div className="flex overflow-x-auto gap-3 pb-1 scrollbar-hide">
                      {results.similar.similar_loans.slice(0, 6).map((loan, i) => (
                        <div key={i} className="min-w-[160px] p-3 bg-slate-50 rounded-xl border border-slate-100 flex-shrink-0">
                          <div className={`text-[10px] font-bold uppercase ${loan.MIS_Status === 'P I F' ? 'text-green-500' : 'text-red-400'}`}>
                            {loan.MIS_Status === 'P I F' ? 'Repaid ✓' : 'Defaulted ✗'}
                          </div>
                          <div className="text-sm font-bold mt-1">${(loan.DisbursementGross || 0).toLocaleString()}</div>
                          <div className="text-[10px] text-slate-400 mt-0.5">{loan.Term}mo · {loan.NoEmp} emp</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
             </div>
          )}
        </div>

      </div>
    </div>
  )
}
