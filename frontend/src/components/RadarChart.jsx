import { ResponsiveContainer, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Tooltip } from "recharts"

export default function AssessmentRadarChart({ features }) {
  // Convert 11 features into 6 normalized dimensions (0-100 scale) for radar plotting
  // Matches the logic in the original Streamlit app
  const scores = [
    {
      subject: 'Loan Term',
      A: Math.min(100, (features.Term / 240) * 100),
      fullMark: 100,
    },
    {
      subject: 'Employment',
      A: Math.min(100, ((features.NoEmp + features.CreateJob + features.RetainedJob) / 30) * 100),
      fullMark: 100,
    },
    {
      subject: 'Maturity',
      A: features.NewExist === 1 ? 80 : 30,
      fullMark: 100,
    },
    {
      subject: 'Guarantee',
      A: Math.min(100, (features.SBA_Appv / Math.max(features.GrAppv, 1)) * 120),
      fullMark: 100,
    },
    {
      subject: 'Location',
      A: features.UrbanRural === 1 ? 80 : (features.UrbanRural === 0 ? 50 : 40),
      fullMark: 100,
    },
    {
      subject: 'Documentation',
      A: Math.max(0, 100 - (features.LowDoc === 1 ? 40 : 0) - (features.RevLineCr === 1 ? 20 : 0)),
      fullMark: 100,
    }
  ]

  return (
    <ResponsiveContainer width="100%" height="100%">
      <RadarChart cx="50%" cy="50%" outerRadius="70%" data={scores}>
        <PolarGrid stroke="var(--border)" />
        <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-secondary)', fontSize: 12, fontWeight: 500 }} />
        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
        <Tooltip 
          contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '8px', color: 'var(--text-primary)' }}
          itemStyle={{ color: 'var(--text-primary)' }}
          formatter={(value) => [`${Math.round(value)}/100`, "Score"]}
        />
        <Radar name="Business Profile" dataKey="A" stroke="var(--accent-primary)" fill="var(--accent-primary)" fillOpacity={0.4} />
      </RadarChart>
    </ResponsiveContainer>
  )
}
