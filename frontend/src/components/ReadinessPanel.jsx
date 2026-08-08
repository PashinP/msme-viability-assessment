import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { 
  ChevronDown, ChevronUp, AlertTriangle, CheckCircle2, 
  Info, TrendingUp, Wrench, Landmark, ExternalLink, Lightbulb, 
  ShieldCheck, BadgeAlert, CircleDashed
} from "lucide-react"

const STATUS_CONFIG = {
  strong:          { color: "#22c55e", bg: "#f0fdf4", border: "#bbf7d0", icon: CheckCircle2,  label: "Strong" },
  moderate:        { color: "#eab308", bg: "#fefce8", border: "#fef08a", icon: Info,           label: "Moderate" },
  needs_attention: { color: "#f97316", bg: "#fff7ed", border: "#fed7aa", icon: AlertTriangle,  label: "Needs Attention" },
  critical:        { color: "#ef4444", bg: "#fef2f2", border: "#fecaca", icon: BadgeAlert,     label: "Critical" },
  unknown:         { color: "#94a3b8", bg: "#f8fafc", border: "#e2e8f0", icon: CircleDashed,   label: "Unknown" },
}

const PRIORITY_BADGE = {
  high:   { bg: "#fef2f2", color: "#dc2626", label: "High Priority" },
  medium: { bg: "#fffbeb", color: "#d97706", label: "Medium" },
  low:    { bg: "#f0fdf4", color: "#16a34a", label: "Low" },
}

// ─── Inline SVG Mini Charts ──────────────────────────────

function GaugeChart({ value, max = 100, label, color, size = 64 }) {
  const pct = Math.min(value / max, 1)
  const angle = pct * 180
  const r = (size - 8) / 2
  const cx = size / 2
  const cy = size / 2 + 4

  const startX = cx - r
  const startY = cy
  const endAngle = (Math.PI * angle) / 180
  const endX = cx - r * Math.cos(endAngle)
  const endY = cy - r * Math.sin(endAngle)
  const largeArc = angle > 180 ? 1 : 0

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size / 2 + 12} viewBox={`0 0 ${size} ${size / 2 + 16}`}>
        {/* Background arc */}
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke="#e2e8f0" strokeWidth="5" strokeLinecap="round" />
        {/* Value arc */}
        {pct > 0 && (
          <path d={`M ${startX} ${startY} A ${r} ${r} 0 ${largeArc} 1 ${endX} ${endY}`} fill="none" stroke={color} strokeWidth="5" strokeLinecap="round" />
        )}
        <text x={cx} y={cy - 4} textAnchor="middle" fontSize="13" fontWeight="700" fill="#1e293b">{typeof value === "number" ? (value > 100 ? `${value.toFixed(0)}` : `${value.toFixed(0)}%`) : value}</text>
      </svg>
      {label && <span className="text-[10px] text-slate-400 text-center leading-tight">{label}</span>}
    </div>
  )
}

function HorizontalBar({ value, max = 10, color, label, height = 6 }) {
  const pct = Math.min(value / max, 1) * 100
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 bg-slate-100 rounded-full overflow-hidden" style={{ height }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-bold text-slate-600 w-8 text-right">{value}/10</span>
    </div>
  )
}

function StackedBar({ segments, height = 24 }) {
  const total = segments.reduce((s, seg) => s + seg.value, 0)
  if (total === 0) return null
  return (
    <div className="w-full">
      <div className="flex rounded-lg overflow-hidden" style={{ height }}>
        {segments.map((seg, i) => {
          const pct = (seg.value / total) * 100
          if (pct < 1) return null
          return (
            <motion.div
              key={i}
              initial={{ width: 0 }}
              animate={{ width: `${pct}%` }}
              transition={{ duration: 0.6, delay: i * 0.1 }}
              style={{ backgroundColor: seg.color }}
              className="flex items-center justify-center"
              title={`${seg.label}: ₹${seg.value.toLocaleString('en-IN')}`}
            >
              {pct > 12 && <span className="text-[9px] font-semibold text-white truncate px-1">{seg.label}</span>}
            </motion.div>
          )
        })}
      </div>
      <div className="flex gap-3 mt-2 flex-wrap">
        {segments.map((seg, i) => (
          <div key={i} className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: seg.color }} />
            <span className="text-[10px] text-slate-500">{seg.label}: ₹{seg.value.toLocaleString('en-IN')}</span>
          </div>
        ))}
      </div>
    </div>
  )
}


// ─── Visual for specific section types ────────────────────

function RepaymentVisual({ numbers }) {
  if (!numbers.monthly_revenue) return null
  const segments = [
    { label: "Expenses", value: numbers.monthly_expenses || 0, color: "#94a3b8" },
    { label: "Existing EMI", value: numbers.existing_emi || 0, color: "#f97316" },
    { label: "New EMI", value: numbers.new_emi_estimate || 0, color: "#ef4444" },
    { label: "Buffer", value: Math.max(0, numbers.disposable_income || 0), color: "#22c55e" },
  ]
  return (
    <div className="mt-3 space-y-3">
      <div className="text-[11px] font-medium text-slate-500">Monthly Revenue Breakdown: ₹{(numbers.monthly_revenue || 0).toLocaleString('en-IN')}</div>
      <StackedBar segments={segments} />
      <div className="flex gap-4 mt-2">
        <GaugeChart value={numbers.dti_ratio || 0} max={100} label="Debt-to-Profit" color={numbers.dti_ratio > 60 ? "#ef4444" : numbers.dti_ratio > 40 ? "#eab308" : "#22c55e"} size={72} />
      </div>
    </div>
  )
}

function CollateralVisual({ numbers }) {
  if (!numbers.loan_amount_inr) return null
  const coverage = numbers.coverage_pct || 0
  return (
    <div className="mt-3 flex items-center gap-4">
      <GaugeChart value={Math.min(coverage, 200)} max={200} label="Coverage" color={coverage >= 100 ? "#22c55e" : coverage >= 50 ? "#eab308" : "#ef4444"} size={72} />
      <div className="text-xs text-slate-500 space-y-1">
        <div>Collateral: <span className="font-semibold text-slate-700">₹{(numbers.collateral_value || 0).toLocaleString('en-IN')}</span></div>
        <div>Loan: <span className="font-semibold text-slate-700">₹{(numbers.loan_amount_inr || 0).toLocaleString('en-IN')}</span></div>
      </div>
    </div>
  )
}


// ─── Section Card (expandable) ────────────────────────────
function SectionCard({ section, index }) {
  const [expanded, setExpanded] = useState(section.status === "critical" || section.status === "needs_attention")
  const config = STATUS_CONFIG[section.status] || STATUS_CONFIG.unknown
  const Icon = config.icon

  return (
    <motion.div 
      initial={{ opacity: 0, y: 12 }} 
      animate={{ opacity: 1, y: 0 }} 
      transition={{ delay: index * 0.08 }}
      style={{ borderColor: config.border }}
      className="border rounded-xl overflow-hidden bg-white"
    >
      {/* Header — always visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-slate-50/50 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0" style={{ backgroundColor: config.bg }}>
            <Icon className="w-4 h-4" style={{ color: config.color }} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-semibold text-sm text-slate-800">{section.section}</div>
            {section.score !== null && (
              <HorizontalBar value={section.score} max={10} color={config.color} />
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 ml-2">
          <span className="text-xs font-bold px-2 py-0.5 rounded-full" style={{ backgroundColor: config.bg, color: config.color }}>
            {config.label}
          </span>
          {expanded ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </div>
      </button>

      {/* Body — expandable */}
      <AnimatePresence>
        {expanded && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }} 
            animate={{ height: "auto", opacity: 1 }} 
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 border-t border-slate-100 pt-3">
              {/* What bank sees — compact */}
              <div className="text-[11px] text-slate-400 mb-2 font-medium tracking-wide uppercase">What a bank sees</div>
              <div className="text-xs text-slate-500 bg-slate-50 px-3 py-2 rounded-lg mb-3">{section.what_bank_sees}</div>
              
              {/* Diagnosis */}
              <div className="text-sm text-slate-600 leading-relaxed">{section.diagnosis}</div>

              {/* Visual charts for specific sections */}
              {section.section === "Repayment Capacity" && section.key_numbers && (
                <RepaymentVisual numbers={section.key_numbers} />
              )}
              {section.section === "Collateral & Security" && section.key_numbers && (
                <CollateralVisual numbers={section.key_numbers} />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}


// ─── Prescription Card ────────────────────────────────────
function PrescriptionCard({ prescription, index }) {
  const [expanded, setExpanded] = useState(index === 0)

  return (
    <motion.div 
      initial={{ opacity: 0, y: 12 }} 
      animate={{ opacity: 1, y: 0 }} 
      transition={{ delay: 0.3 + index * 0.1 }}
      className="border border-blue-100 rounded-xl bg-blue-50/30 overflow-hidden"
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-blue-50/60 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Wrench className="w-4 h-4 text-blue-500" />
          <span className="font-semibold text-sm text-blue-800">
            Fix: {prescription.section}
          </span>
          <span className="text-xs text-blue-400 ml-1">{prescription.suggestions.length} suggestion{prescription.suggestions.length > 1 ? "s" : ""}</span>
        </div>
        {expanded ? <ChevronUp className="w-4 h-4 text-blue-400" /> : <ChevronDown className="w-4 h-4 text-blue-400" />}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }} 
            animate={{ height: "auto", opacity: 1 }} 
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-2">
              {prescription.suggestions.map((suggestion, i) => {
                const priorityCfg = PRIORITY_BADGE[suggestion.priority] || PRIORITY_BADGE.medium
                return (
                  <div key={i} className="bg-white border border-slate-100 rounded-lg p-3">
                    <div className="flex items-start justify-between gap-2 mb-1.5">
                      <div className="flex items-center gap-2">
                        <div className="w-5 h-5 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center font-bold flex-shrink-0">{i + 1}</div>
                        <span className="font-semibold text-sm text-slate-800">{suggestion.action}</span>
                      </div>
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded-full flex-shrink-0" style={{ backgroundColor: priorityCfg.bg, color: priorityCfg.color }}>
                        {priorityCfg.label}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 leading-relaxed ml-7">{suggestion.detail}</p>
                    <div className="ml-7 mt-1.5 flex items-center gap-3 flex-wrap">
                      {suggestion.impact && (
                        <span className="flex items-center gap-1 text-xs text-green-600">
                          <TrendingUp className="w-3 h-3" /> {suggestion.impact}
                        </span>
                      )}
                      {suggestion.difficulty && (
                        <span className="text-[10px] text-slate-400">Difficulty: {suggestion.difficulty}</span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}


// ─── Main Readiness Panel ─────────────────────────────────
export default function ReadinessPanel({ assessmentData }) {
  if (!assessmentData) return null

  const { prediction, assessment, prescriptions, schemes } = assessmentData
  const { sections, overall_status, overall_score, summary, strengths, weaknesses } = assessment

  const statusCfg = STATUS_CONFIG[overall_status] || STATUS_CONFIG.unknown

  // Count sections by status for the overview
  const statusCounts = sections.reduce((acc, s) => {
    acc[s.status] = (acc[s.status] || 0) + 1
    return acc
  }, {})

  return (
    <div className="space-y-5">

      {/* ── Overall Status Bar ── */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.98 }} 
        animate={{ opacity: 1, scale: 1 }}
        className="glass-panel p-5 rounded-2xl"
      >
        <div className="flex items-start gap-5">
          {/* Score Circle */}
          <div className="relative flex-shrink-0">
            <svg width="88" height="88" viewBox="0 0 88 88">
              <circle cx="44" cy="44" r="36" fill="none" stroke="#e2e8f0" strokeWidth="7" />
              <motion.circle 
                cx="44" cy="44" r="36" fill="none" 
                stroke={statusCfg.color} strokeWidth="7"
                strokeDasharray={`${(overall_score / 100) * 226.2} 226.2`}
                strokeLinecap="round"
                initial={{ strokeDasharray: "0 226.2" }}
                animate={{ strokeDasharray: `${(overall_score / 100) * 226.2} 226.2` }}
                transition={{ duration: 1, ease: "easeOut" }}
                transform="rotate(-90 44 44)"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-xl font-bold text-slate-800">{overall_score}</span>
              <span className="text-[10px] text-slate-400">/100</span>
            </div>
          </div>

          {/* Summary Text */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <h2 className="text-base font-bold text-slate-800">Loan Readiness</h2>
              <span className="text-xs font-bold px-2.5 py-0.5 rounded-full" style={{ backgroundColor: statusCfg.bg, color: statusCfg.color }}>
                {statusCfg.label}
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed mb-2">{summary}</p>
            
            {/* Status summary chips */}
            <div className="flex gap-2 flex-wrap">
              {Object.entries(statusCounts).map(([status, count]) => {
                const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.unknown
                return (
                  <span key={status} className="text-[10px] font-medium px-2 py-0.5 rounded-full flex items-center gap-1" style={{ backgroundColor: cfg.bg, color: cfg.color }}>
                    {count} {cfg.label}
                  </span>
                )
              })}
            </div>

            {/* ML Prediction badge */}
            {prediction && (
              <div className="mt-2 flex items-center gap-2 text-[11px] text-slate-400">
                <ShieldCheck className="w-3.5 h-3.5" />
                ML Model: <span className="font-semibold text-slate-600">{prediction.predicted_label}</span> ({(prediction.confidence * 100).toFixed(0)}%)
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* ── Section-by-Section Diagnostic ── */}
      <div>
        <h3 className="text-xs font-semibold text-slate-400 mb-2 flex items-center gap-2 uppercase tracking-wider">
          <Lightbulb className="w-3.5 h-3.5" /> Detailed Diagnosis
        </h3>
        <div className="space-y-2">
          {sections.map((section, i) => (
            <SectionCard key={section.section} section={section} index={i} />
          ))}
        </div>
      </div>

      {/* ── Prescriptions (Actionable Fixes) ── */}
      {prescriptions && prescriptions.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold text-blue-500 mb-2 flex items-center gap-2 uppercase tracking-wider">
            <Wrench className="w-3.5 h-3.5" /> Action Plan
          </h3>
          <div className="space-y-2">
            {prescriptions.map((rx, i) => (
              <PrescriptionCard key={rx.section} prescription={rx} index={i} />
            ))}
          </div>
        </div>
      )}

      {/* ── Government Schemes ── */}
      {schemes && schemes.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold text-slate-400 mb-2 flex items-center gap-2 uppercase tracking-wider">
            <Landmark className="w-3.5 h-3.5" /> Government Schemes
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {schemes.map((scheme, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 + i * 0.06 }}
                className="p-3 bg-white border border-slate-200 rounded-xl"
              >
                <div className="flex justify-between items-start mb-1.5">
                  <span className="font-semibold text-xs text-slate-800">{scheme.name}</span>
                  {scheme.url && (
                    <a href={scheme.url} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-700">
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  )}
                </div>
                <p className="text-[11px] text-slate-500 leading-relaxed mb-1.5">{scheme.description}</p>
                {scheme.benefits && (
                  <div className="flex flex-wrap gap-1">
                    {scheme.benefits.slice(0, 3).map((b, j) => (
                      <span key={j} className="text-[9px] bg-green-50 text-green-700 px-1.5 py-0.5 rounded-full border border-green-100">{b}</span>
                    ))}
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
