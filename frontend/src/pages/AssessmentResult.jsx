import { useState, useEffect } from "react"
import { useLocation, useNavigate } from "react-router-dom"
import { motion } from "framer-motion"
import axios from "axios"
import { CheckCircle, AlertTriangle, ArrowRight, Download, Activity, Landmark } from "lucide-react"
import { Button } from "@/components/ui/button"
import AssessmentRadarChart from "@/components/RadarChart"
import ShapWaterfall from "@/components/ShapWaterfall"

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"
const API_KEY = import.meta.env.VITE_API_KEY || "msme-dev-key-2024"

const GRADE_COLORS = {
  "Critical": "var(--grade-critical)",
  "At-Risk": "var(--grade-atrisk)",
  "Stable": "var(--grade-stable)",
  "Growing": "var(--grade-growing)",
  "Thriving": "var(--grade-thriving)"
}

export default function AssessmentResult() {
  const location = useLocation()
  const navigate = useNavigate()
  const [data, setData] = useState({
    prediction: null,
    explanation: null,
    redflags: null,
    schemes: null,
    similar: null,
    optimization: null
  })
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const features = location.state?.features
    if (!features) {
      navigate("/expert")
      return
    }

    const fetchData = async () => {
      try {
        const headers = { "X-API-Key": API_KEY }
        const [pred, expl, flags, schemes, sim, opt] = await Promise.all([
          axios.post(`${API_URL}/predict`, features, { headers }),
          axios.post(`${API_URL}/explain`, features, { headers }),
          axios.post(`${API_URL}/redflags`, features, { headers }),
          axios.post(`${API_URL}/schemes`, features, { headers }),
          axios.post(`${API_URL}/similar`, features, { headers }),
          axios.post(`${API_URL}/optimize`, features, { headers })
        ])

        setData({
          prediction: pred.data,
          explanation: expl.data,
          redflags: flags.data,
          schemes: schemes.data,
          similar: sim.data,
          optimization: opt.data
        })
      } catch (err) {
        console.error("Error fetching assessment data", err)
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [location, navigate])

  if (isLoading) {
    return (
      <div className="container mx-auto py-20 flex flex-col items-center justify-center space-y-6">
        <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
        <h2 className="text-2xl font-bold animate-pulse">Running Viability Assessment...</h2>
        <p className="text-muted-foreground text-center max-w-md">
          Analyzing 897,000 historical business loans, running XGBoost model, calculating SHAP values, and finding similar peers...
        </p>
      </div>
    )
  }

  const { prediction } = data
  if (!prediction) return <div className="p-8">Error loading data.</div>

  return (
    <div className="container mx-auto py-8 space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Your Loan Readiness Report</h1>
          <p className="text-muted-foreground">Comprehensive viability analysis and prescriptive interventions.</p>
        </div>
        <Button className="gap-2">
          <Download className="w-4 h-4" /> Download PDF
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Grade & Probability */}
        <div className="lg:col-span-1 space-y-6">
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="glass-panel p-8 text-center rounded-3xl relative overflow-hidden"
          >
            <div 
              className="absolute inset-0 opacity-20"
              style={{ backgroundColor: GRADE_COLORS[prediction.predicted_label] }}
            />
            <div className="relative z-10 flex flex-col items-center">
              <h3 className="text-xl font-medium mb-4 text-white/80">Viability Grade</h3>
              <div 
                className="w-32 h-32 rounded-2xl flex items-center justify-center mb-4 text-7xl font-black shadow-2xl"
                style={{ backgroundColor: GRADE_COLORS[prediction.predicted_label], color: "white" }}
              >
                {prediction.predicted_label[0]}
              </div>
              <h2 className="text-3xl font-bold">{prediction.predicted_label}</h2>
              <p className="mt-2 text-lg font-medium text-white/90">
                {(prediction.confidence * 100).toFixed(1)}% Confidence
              </p>
            </div>
          </motion.div>

          <div className="glass-panel p-6 rounded-2xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-accent-cyan" /> 
              Class Probabilities
            </h3>
            <div className="space-y-4">
              {Object.entries(prediction.probabilities).map(([label, prob], i) => (
                <div key={label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="font-medium text-muted-foreground">{label}</span>
                    <span className="font-bold">{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${prob * 100}%` }}
                      transition={{ duration: 0.8, delay: i * 0.1 }}
                      className="h-full rounded-full"
                      style={{ backgroundColor: GRADE_COLORS[label] }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: SHAP, Radar, Recommendations */}
        <div className="lg:col-span-2 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass-panel p-6 rounded-2xl h-80 flex flex-col">
              <h4 className="text-sm font-semibold mb-2 text-muted-foreground text-center">Business Profile</h4>
              <div className="flex-1">
                <AssessmentRadarChart features={location.state.features} />
              </div>
            </div>
            <div className="glass-panel p-6 rounded-2xl h-80 flex items-center justify-center">
              <ShapWaterfall explanation={data.explanation} />
            </div>
          </div>

          <div className="glass-panel p-6 rounded-2xl">
            <h3 className="text-lg font-semibold mb-4 text-accent-cyan flex items-center gap-2">
              <CheckCircle className="w-5 h-5" /> 
              How to Improve Your Application
            </h3>
            {data.optimization?.changes && data.optimization.changes.length > 0 ? (
              <div className="space-y-4">
                <div className="bg-primary/20 text-primary-foreground p-4 rounded-lg font-medium">
                  By restructuring your loan, you can upgrade from <span className="font-bold">{data.optimization.original_prediction.predicted_label}</span> to <span className="font-bold">{data.optimization.optimized_prediction.predicted_label}</span>!
                </div>
                {data.optimization.changes.map((change, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 bg-secondary rounded-lg">
                    <ArrowRight className="w-5 h-5 text-accent-purple" />
                    <div>
                      Change <span className="font-medium">{change.feature}</span> from <span className="text-red-400">{change.original}</span> to <span className="text-green-400">{change.optimized}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground">Your loan is already optimally structured for its class.</p>
            )}
          </div>

          {data.redflags?.flags?.length > 0 && (
            <div className="glass-panel p-6 rounded-2xl border-red-500/30">
              <h3 className="text-lg font-semibold mb-4 text-red-400 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" /> 
                Risk Red Flags Detected
              </h3>
              <div className="space-y-3">
                {data.redflags.flags.map((flag, i) => (
                  <div key={i} className="p-4 bg-red-500/10 rounded-lg border border-red-500/20">
                    <div className="font-bold text-red-300 mb-1">{flag.flag}</div>
                    <div className="text-sm text-red-200/70">{flag.explanation}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {data.schemes?.matches?.length > 0 && (
            <div className="glass-panel p-6 rounded-2xl">
              <h3 className="text-lg font-semibold mb-4 text-accent-blue flex items-center gap-2">
                <Landmark className="w-5 h-5" /> 
                Recommended Government Schemes
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {data.schemes.matches.map((scheme, i) => (
                  <div key={i} className="p-4 bg-accent-blue/10 rounded-xl border border-accent-blue/20">
                    <div className="font-bold text-accent-blue mb-1 flex justify-between">
                      {scheme.scheme}
                      <span className="text-xs bg-accent-blue/20 px-2 py-0.5 rounded-full">Match: {(scheme.match_score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="text-sm text-foreground/80 mb-2">{scheme.description}</div>
                    <div className="text-xs text-muted-foreground font-mono">Limit: {scheme.max_amount} | {scheme.interest_rate}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}
