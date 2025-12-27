import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import data from '../data/insights.json'

export default function Environment() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🌍 Environmental Impact</h1>
                <p className="page-subtitle">Lifecycle emissions, waste toxicity, and grid requirements</p>
            </header>

            <div className="grid-2">
                <div className="chart-container">
                    <h3 className="chart-title">🚛 Transport Efficiency (CO2 per ton-mile)</h3>
                    <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={data.transport.efficiency} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis type="number" stroke="#71717a" />
                            <YAxis dataKey="mode" type="category" stroke="#71717a" width={60} />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                            <Bar dataKey="co2PerTonMile" radius={[0, 8, 8, 0]}>
                                <Cell fill="#22c55e" />
                                <Cell fill="#3b82f6" />
                                <Cell fill="#f97316" />
                                <Cell fill="#ef4444" />
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.5rem' }}>
                        Ships are 82x more efficient than planes for freight!
                    </p>
                </div>

                <div className="chart-container">
                    <h3 className="chart-title">⚡ Lifecycle Emissions (150K miles)</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
                        <div className="card" style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Gas Car</div>
                            <div style={{ fontSize: '2.5rem', fontWeight: 700, color: '#ef4444' }}>67.3</div>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>tonnes CO2</div>
                        </div>
                        <div className="card" style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>EV</div>
                            <div style={{ fontSize: '2.5rem', fontWeight: 700, color: '#22c55e' }}>28.9</div>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>tonnes CO2</div>
                        </div>
                    </div>
                    <p style={{ color: 'var(--accent-green)', fontWeight: 600, marginTop: '1rem', textAlign: 'center' }}>
                        57% less CO2 over vehicle lifetime!
                    </p>
                </div>
            </div>

            <div className="chart-container">
                <h3 className="chart-title">📊 Environmental Key Findings</h3>
                <div className="grid-3" style={{ marginTop: '1rem' }}>
                    <div className="card">
                        <h4 style={{ color: 'var(--accent-green)' }}>🔄 Carbon Payback</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                            EVs "pay back" manufacturing CO2 in just <strong>2 years</strong> or 25,000 miles
                        </p>
                    </div>
                    <div className="card">
                        <h4 style={{ color: 'var(--accent-blue)' }}>⚡ Grid Impact</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                            All-EV future needs only <strong>21% more</strong> electricity
                        </p>
                    </div>
                    <div className="card">
                        <h4 style={{ color: 'var(--accent-purple)' }}>♻️ Battery Recycling</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                            <strong>95% recyclable</strong> - valuable metals recovered
                        </p>
                    </div>
                    <div className="card">
                        <h4 style={{ color: 'var(--accent-orange)' }}>🛢️ Motor Oil</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                            1 gallon contaminates <strong>1 million gallons</strong> of water
                        </p>
                    </div>
                    <div className="card">
                        <h4 style={{ color: 'var(--accent-cyan)' }}>☀️ Clean Grid</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                            On solar/wind: EVs are <strong>73% cleaner</strong>
                        </p>
                    </div>
                    <div className="card">
                        <h4 style={{ color: 'var(--accent-green)' }}>🗑️ Waste Comparison</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                            ICE waste is <strong>3x more dangerous</strong> than EV waste
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
