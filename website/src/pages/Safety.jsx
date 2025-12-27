import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import data from '../data/insights.json'

export default function Safety() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🔒 Safety Statistics</h1>
                <p className="page-subtitle">Fire rates, crash data, and injury statistics</p>
            </header>

            <div className="grid-2">
                <div className="chart-container">
                    <h3 className="chart-title">🔥 Fire Risk (per 100,000 vehicles)</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={data.safety.fireRates} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis type="number" stroke="#71717a" />
                            <YAxis dataKey="type" type="category" stroke="#71717a" width={100} />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                            <Bar dataKey="rate" radius={[0, 8, 8, 0]}>
                                {data.safety.fireRates.map((entry, i) => (
                                    <Cell key={i} fill={entry.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                    <p style={{ color: 'var(--accent-green)', fontWeight: 600, marginTop: '1rem', textAlign: 'center' }}>
                        EVs have 98% fewer fires than gas cars!
                    </p>
                </div>

                <div className="chart-container">
                    <h3 className="chart-title">📊 Key Safety Findings</h3>
                    <div style={{ display: 'grid', gap: '1rem', marginTop: '1rem' }}>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-green)' }}>
                            <h4>🔥 Fire Risk</h4>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                                Gas: 1,530 • Hybrid: 500 • <strong style={{ color: 'var(--accent-green)' }}>EV: 25</strong>
                            </p>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-blue)' }}>
                            <h4>🚗 Injury Claims</h4>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                                EVs have <strong style={{ color: 'var(--accent-blue)' }}>40% fewer</strong> injury claims (IIHS)
                            </p>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-purple)' }}>
                            <h4>🤖 Tesla Autopilot</h4>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                                <strong style={{ color: 'var(--accent-purple)' }}>7.6M miles</strong> per crash (vs 0.67M national avg)
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="chart-container">
                <h3 className="chart-title">Why EVs Are Safer</h3>
                <div className="grid-3" style={{ marginTop: '1rem' }}>
                    <div className="card">
                        <h4>⬇️ Lower Center of Gravity</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                            Battery in floor = 30% reduced rollover risk
                        </p>
                    </div>
                    <div className="card">
                        <h4>🛡️ Better Crumple Zones</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                            No engine = more space for impact absorption
                        </p>
                    </div>
                    <div className="card">
                        <h4>⚖️ Heavier Weight</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                            Physics: heavier vehicles protect occupants better
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
