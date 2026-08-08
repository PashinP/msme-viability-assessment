import { useState, useEffect, useCallback } from "react"
import axios from "axios"
import { motion } from "framer-motion"
import { ArrowRight, Activity, Settings2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"
const API_KEY = import.meta.env.VITE_API_KEY || "msme-dev-key-2024"

const GRADE_COLORS = {
  "Critical": "var(--grade-critical)",
  "At-Risk": "var(--grade-atrisk)",
  "Stable": "var(--grade-stable)",
  "Growing": "var(--grade-growing)",
  "Thriving": "var(--grade-thriving)"
}

export default function ExpertMode() {
  const [features, setFeatures] = useState({
    Term: 84,
    NoEmp: 5,
    NewExist: 1,
    CreateJob: 2,
    RetainedJob: 5,
    DisbursementGross: 100000,
    UrbanRural: 1,
    RevLineCr: 0,
    LowDoc: 0,
    SBA_Appv: 75000,
    GrAppv: 100000
  })

  const [result, setResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  // Debounce API call
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchPrediction()
    }, 500)
    return () => clearTimeout(timer)
  }, [features])

  const fetchPrediction = async () => {
    setIsLoading(true)
    try {
      const { data } = await axios.post(`${API_URL}/predict`, features, {
        headers: { "X-API-Key": API_KEY }
      })
      setResult(data)
    } catch (err) {
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleChange = (e) => {
    const { name, value, type } = e.target
    setFeatures(prev => ({
      ...prev,
      [name]: type === 'number' || type === 'range' ? Number(value) : Number(value)
    }))
  }

  return (
    <div className="container mx-auto py-8">
      <div className="flex items-center gap-3 mb-8">
        <div className="w-10 h-10 rounded-xl bg-accent-blue/20 flex items-center justify-center">
          <Settings2 className="w-5 h-5 text-accent-blue" />
        </div>
        <div>
          <h1 className="text-3xl font-bold">Expert Mode</h1>
          <p className="text-muted-foreground">Adjust parameters and see live viability scoring.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Left side: Controls */}
        <div className="space-y-6">
          <Card className="bg-secondary border-none">
            <CardHeader>
              <CardTitle className="text-lg">Loan Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">Loan Term (Months): {features.Term}</label>
                </div>
                <input type="range" name="Term" min="12" max="240" step="12" value={features.Term} onChange={handleChange} className="w-full accent-primary" />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">Disbursement Amount ($): ${features.DisbursementGross.toLocaleString()}</label>
                </div>
                <input type="range" name="DisbursementGross" min="10000" max="5000000" step="10000" value={features.DisbursementGross} onChange={handleChange} className="w-full accent-primary" />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">SBA Guarantee ($): ${features.SBA_Appv.toLocaleString()}</label>
                </div>
                <input type="range" name="SBA_Appv" min="0" max="5000000" step="10000" value={features.SBA_Appv} onChange={handleChange} className="w-full accent-primary" />
              </div>

              <div className="grid grid-cols-2 gap-4 pt-2">
                <div className="space-y-1">
                  <label className="text-sm text-muted-foreground">Employees</label>
                  <input type="number" name="NoEmp" value={features.NoEmp} onChange={handleChange} className="w-full bg-background border border-border rounded-md px-3 py-2" />
                </div>
                <div className="space-y-1">
                  <label className="text-sm text-muted-foreground">Business Type</label>
                  <select name="NewExist" value={features.NewExist} onChange={handleChange} className="w-full bg-background border border-border rounded-md px-3 py-2">
                    <option value={1}>Existing</option>
                    <option value={2}>New Startup</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-sm text-muted-foreground">Location</label>
                  <select name="UrbanRural" value={features.UrbanRural} onChange={handleChange} className="w-full bg-background border border-border rounded-md px-3 py-2">
                    <option value={1}>Urban</option>
                    <option value={2}>Rural</option>
                    <option value={0}>Undefined</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-sm text-muted-foreground">Low Doc Loan</label>
                  <select name="LowDoc" value={features.LowDoc} onChange={handleChange} className="w-full bg-background border border-border rounded-md px-3 py-2">
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </div>
              </div>

            </CardContent>
          </Card>
        </div>

        {/* Right side: Results */}
        <div>
          <div className="sticky top-24 space-y-6">
            {!result ? (
              <div className="glass-panel p-12 text-center flex flex-col items-center">
                <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4" />
                <p className="text-muted-foreground">Calculating viability...</p>
              </div>
            ) : (
              <>
                <motion.div 
                  key={result.predicted_label} // animate when label changes
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-panel p-8 text-center rounded-3xl relative overflow-hidden transition-colors duration-500"
                  style={{ borderColor: GRADE_COLORS[result.predicted_label] }}
                >
                  <div 
                    className="absolute inset-0 opacity-10 transition-colors duration-500"
                    style={{ backgroundColor: GRADE_COLORS[result.predicted_label] }}
                  />
                  <div className="relative z-10 flex flex-col items-center">
                    <h3 className="text-xl font-medium mb-4 text-white/80">Live Viability Grade</h3>
                    <div 
                      className="w-24 h-24 rounded-2xl flex items-center justify-center mb-4 text-5xl font-black shadow-lg transition-colors duration-500"
                      style={{ backgroundColor: GRADE_COLORS[result.predicted_label], color: "white" }}
                    >
                      {result.predicted_label[0]}
                    </div>
                    <h2 className="text-3xl font-bold" style={{ color: GRADE_COLORS[result.predicted_label] }}>
                      {result.predicted_label}
                    </h2>
                    <p className="mt-2 font-medium">
                      {(result.confidence * 100).toFixed(1)}% Confidence
                    </p>
                  </div>
                </motion.div>

                <div className="glass-panel p-6 rounded-2xl">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-accent-cyan" /> 
                    Probabilities
                  </h3>
                  <div className="space-y-4">
                    {Object.entries(result.probabilities).map(([label, prob]) => (
                      <div key={label}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-muted-foreground">{label}</span>
                          <span className="font-bold">{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: `${prob * 100}%` }}
                            transition={{ duration: 0.5 }}
                            className="h-full rounded-full"
                            style={{ backgroundColor: GRADE_COLORS[label] }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

      </div>
    </div>
  )
}
