import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts'
import ChartModal from '../components/ChartModal'
import data from '../data/insights.json'

export default function MarketInsights() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">📊 2024 Market Insights</h1>
                <p className="page-subtitle">Latest manufacturer rankings, solid-state batteries, and market trends</p>
            </header>

            {/* 2024 Market Share */}
            <div className="grid-2">
                <ChartModal
                    title="🏆 2024 Global EV Sales (Millions)"
                    insight={data.marketShare2024.keyInsight}
                >
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data.marketShare2024.manufacturers} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis type="number" stroke="#71717a" />
                            <YAxis dataKey="name" type="category" stroke="#71717a" width={80} />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                            <Bar dataKey="sales" radius={[0, 8, 8, 0]}>
                                <Cell fill="#22c55e" />
                                <Cell fill="#ef4444" />
                                <Cell fill="#3b82f6" />
                                <Cell fill="#f97316" />
                                <Cell fill="#a855f7" />
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </ChartModal>

                <div className="chart-container">
                    <h3 className="chart-title">📈 YoY Growth Rate</h3>
                    <div style={{ display: 'grid', gap: '0.5rem', marginTop: '0.5rem' }}>
                        {data.marketShare2024.manufacturers.map((m, i) => (
                            <div key={i} className="card" style={{ padding: '0.75rem', display: 'flex', justifyContent: 'space-between' }}>
                                <span>{m.name}</span>
                                <span style={{
                                    color: m.change >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
                                    fontWeight: 600
                                }}>
                                    {m.change >= 0 ? '+' : ''}{m.change}%
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Solid State Battery Timeline */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🔋 Solid-State Battery Timeline</h3>
                <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
                    Next-gen batteries: 2x energy density, 10-min charging, no fire risk
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.5rem', marginTop: '1rem' }}>
                    {data.solidStateBattery.timeline.map((t, i) => (
                        <div key={i} className="card" style={{
                            padding: '1rem',
                            textAlign: 'center',
                            borderTop: `3px solid ${t.status === 'mature' ? 'var(--accent-green)' : t.status === 'growth' ? 'var(--accent-blue)' : 'var(--accent-orange)'}`
                        }}>
                            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-green)' }}>{t.year}</div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>{t.milestone}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Solid State Benefits */}
            <div className="grid-3" style={{ marginTop: '1rem' }}>
                <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-green)' }}>
                    <h4>⚡ Energy Density</h4>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-green)' }}>
                        {data.solidStateBattery.benefits.energyDensity}
                    </div>
                </div>
                <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-blue)' }}>
                    <h4>🔌 Charging Time</h4>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-blue)' }}>
                        {data.solidStateBattery.benefits.chargingTime}
                    </div>
                </div>
                <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-purple)' }}>
                    <h4>🔒 Safety</h4>
                    <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--accent-purple)' }}>
                        {data.solidStateBattery.benefits.safetyImprovement}
                    </div>
                </div>
            </div>

            {/* Winter Range */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">❄️ Winter Range Loss by Model</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <div>
                        <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Best Performers (Range Retention)</h4>
                        {data.winterRange.modelPerformance.filter(m => m.tier === 'best').map((m, i) => (
                            <div key={i} className="card" style={{ padding: '0.5rem 1rem', marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between' }}>
                                <span>{m.model}</span>
                                <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>{m.retention}%</span>
                            </div>
                        ))}
                    </div>
                    <div>
                        <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Needs Improvement</h4>
                        {data.winterRange.modelPerformance.filter(m => m.tier === 'poor').map((m, i) => (
                            <div key={i} className="card" style={{ padding: '0.5rem 1rem', marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between' }}>
                                <span>{m.model}</span>
                                <span style={{ color: 'var(--accent-orange)', fontWeight: 600 }}>{m.retention}%</span>
                            </div>
                        ))}
                    </div>
                </div>
                <div className="grid-3" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '0.75rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Heat Pump</div>
                        <div style={{ fontWeight: 600, color: 'var(--accent-green)' }}>+{data.winterRange.mitigation.heatPump}</div>
                    </div>
                    <div className="card" style={{ padding: '0.75rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Pre-conditioning</div>
                        <div style={{ fontWeight: 600, color: 'var(--accent-blue)' }}>+{data.winterRange.mitigation.preconditioning}</div>
                    </div>
                    <div className="card" style={{ padding: '0.75rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Seat Heaters</div>
                        <div style={{ fontWeight: 600, color: 'var(--accent-purple)' }}>{data.winterRange.mitigation.seatHeaters}</div>
                    </div>
                </div>
            </div>
        </div>
    )
}
