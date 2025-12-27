import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import ChartModal from '../components/ChartModal'
import data from '../data/insights.json'

export default function Safety() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🔒 Safety Statistics</h1>
                <p className="page-subtitle">Fire rates, crash data, and injury statistics</p>
            </header>

            <div className="grid-2">
                <ChartModal
                    title="🔥 Fire Risk (per 100,000 vehicles)"
                    insight="EVs have a dramatically lower fire risk than gas cars: only 25 fires per 100,000 EVs vs 1,530 for gas cars. This is a 98% reduction! EV battery fires, while intense, are extremely rare. The main risk factor for any vehicle fire is age and maintenance."
                >
                    <ResponsiveContainer width="100%" height="100%">
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
                </ChartModal>

                <div className="chart-container">
                    <h3 className="chart-title">📊 Key Safety Findings</h3>
                    <div style={{ display: 'grid', gap: '0.75rem', marginTop: '0.5rem' }}>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-green)', padding: '0.75rem' }}>
                            <h4 style={{ fontSize: '0.9rem' }}>🔥 Fire Risk</h4>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                                Gas: 1,530 • Hybrid: 500 • <strong style={{ color: 'var(--accent-green)' }}>EV: 25</strong>
                            </p>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-blue)', padding: '0.75rem' }}>
                            <h4 style={{ fontSize: '0.9rem' }}>🚗 Injury Claims</h4>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                                EVs have <strong style={{ color: 'var(--accent-blue)' }}>40% fewer</strong> injury claims
                            </p>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-purple)', padding: '0.75rem' }}>
                            <h4 style={{ fontSize: '0.9rem' }}>🤖 Tesla Autopilot</h4>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                                <strong style={{ color: 'var(--accent-purple)' }}>7.6M mi</strong>/crash vs 0.67M avg
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="chart-container">
                <h3 className="chart-title">Why EVs Are Safer</h3>
                <div className="grid-3" style={{ marginTop: '0.75rem' }}>
                    <div className="card" style={{ padding: '1rem' }}>
                        <h4 style={{ fontSize: '0.95rem' }}>⬇️ Lower Center of Gravity</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.85rem' }}>
                            Battery in floor = 30% reduced rollover risk
                        </p>
                    </div>
                    <div className="card" style={{ padding: '1rem' }}>
                        <h4 style={{ fontSize: '0.95rem' }}>🛡️ Better Crumple Zones</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.85rem' }}>
                            No engine = more space for impact absorption
                        </p>
                    </div>
                    <div className="card" style={{ padding: '1rem' }}>
                        <h4 style={{ fontSize: '0.95rem' }}>⚖️ Heavier Weight</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.85rem' }}>
                            Physics: heavier vehicles protect occupants
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
