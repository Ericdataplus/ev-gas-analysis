import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import ChartModal from '../components/ChartModal'
import data from '../data/insights.json'

export default function Semis() {
    const marketGrowth = [
        { year: 2024, value: 0.85 },
        { year: 2026, value: 1.5 },
        { year: 2028, value: 2.8 },
        { year: 2030, value: 4.0 },
        { year: 2033, value: 6.65 },
    ]

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🚛 Electric Semi Trucks</h1>
                <p className="page-subtitle">Tesla Semi, Nikola, and the electrification of freight</p>
            </header>

            {/* Key Stats */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon">📈</div>
                    <div className="stat-value">{data.semiTrucks.market2024.cagr}%</div>
                    <div className="stat-label">Market CAGR</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">💰</div>
                    <div className="stat-value">${(data.semiTrucks.market2024.projection2033 / 1000).toFixed(1)}B</div>
                    <div className="stat-label">Market by 2033</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🚚</div>
                    <div className="stat-value">{data.semiTrucks.teslaSemi.delivered}</div>
                    <div className="stat-label">Tesla Semis Delivered</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">⛽</div>
                    <div className="stat-value">{data.semiTrucks.nikola.fcevsSold2024}</div>
                    <div className="stat-label">Nikola FCEVs Sold</div>
                </div>
            </div>

            <div className="grid-2">
                <ChartModal
                    title="📈 Electric Semi Market Growth ($B)"
                    insight={`Only ${data.semiTrucks.market2024.electricShareOfNewSales}% of new truck sales are electric today, but the market is projected to grow from $${(data.semiTrucks.market2024.valueMillion / 1000).toFixed(1)}B in 2024 to $${(data.semiTrucks.market2024.projection2033 / 1000).toFixed(1)}B by 2033 - a ${data.semiTrucks.market2024.cagr}% CAGR.`}
                >
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={marketGrowth}>
                            <defs>
                                <linearGradient id="semiGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis dataKey="year" stroke="#71717a" />
                            <YAxis stroke="#71717a" unit="B" />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                            <Area type="monotone" dataKey="value" stroke="#22c55e" fill="url(#semiGrad)" strokeWidth={2} />
                        </AreaChart>
                    </ResponsiveContainer>
                </ChartModal>

                {/* Tesla vs Nikola comparison */}
                <div className="chart-container">
                    <h3 className="chart-title">🏭 Market Leaders</h3>
                    <div style={{ display: 'grid', gap: '1rem', marginTop: '0.5rem' }}>
                        {/* Tesla */}
                        <div className="card" style={{ padding: '1rem', borderLeft: '3px solid #ef4444' }}>
                            <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <span style={{ fontSize: '1.5rem' }}>🔴</span> Tesla Semi
                            </h4>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.5rem', fontSize: '0.85rem' }}>
                                <div>Delivered: <strong>{data.semiTrucks.teslaSemi.delivered}</strong></div>
                                <div>Range: <strong>{data.semiTrucks.teslaSemi.range} mi</strong></div>
                                <div>High-Volume: <strong>{data.semiTrucks.teslaSemi.highVolumeProductionStart}</strong></div>
                                <div>Factory Cap: <strong>{(data.semiTrucks.teslaSemi.factoryCapacity / 1000).toFixed(0)}K/yr</strong></div>
                            </div>
                        </div>
                        {/* Nikola */}
                        <div className="card" style={{ padding: '1rem', borderLeft: '3px solid #3b82f6' }}>
                            <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <span style={{ fontSize: '1.5rem' }}>🔵</span> Nikola FCEV
                            </h4>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.5rem', fontSize: '0.85rem' }}>
                                <div>Sold 2024: <strong>{data.semiTrucks.nikola.fcevsSold2024}</strong></div>
                                <div>Fuel: <strong>{data.semiTrucks.nikola.fuelType}</strong></div>
                                <div>Fleet Growth: <strong>+{data.semiTrucks.nikola.fleetAdoptionGrowth}%</strong></div>
                                <div>Guidance: <strong>{data.semiTrucks.nikola.guidance2024.min}-{data.semiTrucks.nikola.guidance2024.max}</strong></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Benefits vs Challenges */}
            <div className="grid-2" style={{ marginTop: '1.5rem' }}>
                <div className="chart-container">
                    <h3 className="chart-title" style={{ color: 'var(--accent-green)' }}>✅ Benefits</h3>
                    <div style={{ display: 'grid', gap: '0.5rem', marginTop: '0.75rem' }}>
                        {data.semiTrucks.benefits.map((b, i) => (
                            <div key={i} className="card" style={{ padding: '0.75rem', borderLeft: '2px solid var(--accent-green)' }}>
                                {b}
                            </div>
                        ))}
                    </div>
                </div>
                <div className="chart-container">
                    <h3 className="chart-title" style={{ color: 'var(--accent-red)' }}>⚠️ Challenges</h3>
                    <div style={{ display: 'grid', gap: '0.5rem', marginTop: '0.75rem' }}>
                        {data.semiTrucks.challenges.map((c, i) => (
                            <div key={i} className="card" style={{ padding: '0.75rem', borderLeft: '2px solid var(--accent-red)' }}>
                                {c}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Future Outlook */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🔮 The Big Picture</h3>
                <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', lineHeight: 1.6 }}>
                    Trucking accounts for <strong>24% of US transportation emissions</strong>. While only 0.1% of new trucks are electric today,
                    the market is growing 25%+ annually. Tesla's 50,000-unit/year factory coming in 2026 could be a game-changer.
                    Meanwhile, Nikola is proving hydrogen fuel cells work for long-haul routes where battery weight is a concern.
                </p>
                <div className="grid-3" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-green)' }}>80%</div>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Fuel Cost Savings</div>
                    </div>
                    <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-blue)' }}>50%</div>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Lower Maintenance</div>
                    </div>
                    <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-purple)' }}>24%</div>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>of Transport Emissions</div>
                    </div>
                </div>
            </div>
        </div>
    )
}
