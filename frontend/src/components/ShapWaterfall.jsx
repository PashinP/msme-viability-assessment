import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ReferenceLine } from "recharts"

export default function ShapWaterfall({ explanation }) {
  if (!explanation || !explanation.feature_contributions) return null

  // Process data for waterfall/bar chart
  const data = Object.entries(explanation.feature_contributions)
    .map(([feature, value]) => ({
      feature,
      value,
      isPositive: value > 0
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 7) // Show top 7 contributors

  return (
    <div className="w-full h-full flex flex-col">
      <h4 className="text-sm font-semibold mb-4 text-muted-foreground text-center">Feature Impact on Grade</h4>
      <div className="flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 30, left: 60, bottom: 0 }}>
            <XAxis type="number" stroke="var(--border)" tick={{ fill: 'var(--text-muted)' }} />
            <YAxis dataKey="feature" type="category" stroke="var(--border)" tick={{ fill: 'var(--text-secondary)', fontSize: 11, fontWeight: 600 }} width={120} />
            <Tooltip 
              cursor={{ fill: 'var(--bg-tertiary)' }}
              contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '8px', zIndex: 50, color: 'var(--text-primary)' }}
              itemStyle={{ color: 'var(--text-primary)' }}
              formatter={(value, name, props) => {
                const isPos = value > 0
                return [
                  <span className={isPos ? "text-green-600 font-medium" : "text-red-600 font-medium"}>
                    {isPos ? 'Pushed grade UP' : 'Pushed grade DOWN'}
                  </span>,
                  "Impact"
                ]
              }}
            />
            <ReferenceLine x={0} stroke="var(--border)" />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.isPositive ? '#22c55e' : '#ef4444'} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
