import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import ChartModal from '../components/ChartModal'
import data from '../data/insights.json'

export default function Environment() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🌍 Environmental Impact</h1>
                <p className="page-subtitle">Lifecycle emissions, waste toxicity, and grid requirements</p>
            </header>

            <div className="grid-2">
                <ChartModal
                    title="🚛 Freight Emissions (CO2 lbs/ton-mile) - Lower is Better"
                    insight="Ships are 82x CLEANER than planes for freight! Air cargo produces 1.23 lbs CO2 per ton-mile vs just 0.015 for ships. This is why most global trade uses shipping. Trucking (0.15) is 10x worse than ships - a key reason for freight rail investment."
                >
                    <ResponsiveContainer width="100%" height="100%">
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
                </ChartModal>

                <div className="chart-container">
                    <h3 className="chart-title">⚡ Lifecycle Emissions (150K mi)</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.5rem' }}>
                        <div className="card" style={{ textAlign: 'center', padding: '1rem' }}>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Gas Car</div>
                            <div style={{ fontSize: '2rem', fontWeight: 700, color: '#ef4444' }}>67.3</div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>tonnes CO2</div>
                        </div>
                        <div className="card" style={{ textAlign: 'center', padding: '1rem' }}>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>EV</div>
                            <div style={{ fontSize: '2rem', fontWeight: 700, color: '#22c55e' }}>28.9</div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>tonnes CO2</div>
                        </div>
                    </div>
                    <p style={{ color: 'var(--accent-green)', fontWeight: 600, marginTop: '0.75rem', textAlign: 'center', fontSize: '0.9rem' }}>
                        57% less CO2 over vehicle lifetime!
                    </p>
                </div>
            </div>

            <div className="chart-container">
                <h3 className="chart-title">📊 Environmental Key Findings</h3>
                <div className="grid-3" style={{ marginTop: '0.75rem' }}>
                    <div className="card" style={{ padding: '0.75rem' }}>
                        <h4 style={{ color: 'var(--accent-green)', fontSize: '0.9rem' }}>🔄 Carbon Payback</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.8rem' }}>
                            EVs "pay back" manufacturing CO2 in <strong>2 years</strong>
                        </p>
                    </div>
                    <div className="card" style={{ padding: '0.75rem' }}>
                        <h4 style={{ color: 'var(--accent-blue)', fontSize: '0.9rem' }}>⚡ Grid Impact</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.8rem' }}>
                            All-EV future needs only <strong>21% more</strong> electricity
                        </p>
                    </div>
                    <div className="card" style={{ padding: '0.75rem' }}>
                        <h4 style={{ color: 'var(--accent-purple)', fontSize: '0.9rem' }}>♻️ Battery Recycling</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.8rem' }}>
                            <strong>95% recyclable</strong> - metals recovered
                        </p>
                    </div>
                    <div className="card" style={{ padding: '0.75rem' }}>
                        <h4 style={{ color: 'var(--accent-orange)', fontSize: '0.9rem' }}>🛢️ Motor Oil</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.8rem' }}>
                            1 gal contaminates <strong>1M gallons</strong> water
                        </p>
                    </div>
                    <div className="card" style={{ padding: '0.75rem' }}>
                        <h4 style={{ color: 'var(--accent-cyan)', fontSize: '0.9rem' }}>☀️ Clean Grid</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.8rem' }}>
                            On solar/wind: EVs are <strong>73% cleaner</strong>
                        </p>
                    </div>
                    <div className="card" style={{ padding: '0.75rem' }}>
                        <h4 style={{ color: 'var(--accent-green)', fontSize: '0.9rem' }}>🗑️ Waste</h4>
                        <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem', fontSize: '0.8rem' }}>
                            ICE waste is <strong>3x more dangerous</strong>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
