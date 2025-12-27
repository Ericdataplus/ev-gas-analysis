import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import ChartModal from '../components/ChartModal'
import data from '../data/insights.json'

export default function UsedEVs() {
    const priceDrops = [
        { model: 'Tesla Model 3', drop: data.usedEVMarket.priceDrops2024.teslaModel3 },
        { model: 'Kia Niro EV', drop: data.usedEVMarket.priceDrops2024.kiaNiroEV },
        { model: 'Nissan Leaf', drop: data.usedEVMarket.priceDrops2024.nissanLeaf },
        { model: 'Hyundai Kona', drop: data.usedEVMarket.priceDrops2024.hyundaiKona },
        { model: 'Chevy Bolt', drop: data.usedEVMarket.priceDrops2024.chevyBolt },
    ]

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🚗 Used EV Market</h1>
                <p className="page-subtitle">2024 pricing trends, affordability, and buying opportunities</p>
            </header>

            {/* Key Stats */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon">📉</div>
                    <div className="stat-value">{data.usedEVMarket.priceDrops2024.average}%</div>
                    <div className="stat-label">Avg Price Drop YoY</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">💰</div>
                    <div className="stat-value">${(data.usedEVMarket.marketGrowth.averagePrice / 1000).toFixed(0)}K</div>
                    <div className="stat-label">Avg Used EV Price</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">📈</div>
                    <div className="stat-value">{data.usedEVMarket.marketGrowth.salesGrowth}%</div>
                    <div className="stat-label">Sales Growth</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">⏱️</div>
                    <div className="stat-value">{data.usedEVMarket.marketGrowth.daysToSell}</div>
                    <div className="stat-label">Avg Days to Sell</div>
                </div>
            </div>

            {/* Price Drops Chart */}
            <ChartModal
                title="📉 2024 Used EV Price Drops by Model"
                insight="Used EVs dropped 29.5% on average in 2024 - 6x faster than gas cars. Tesla Model 3 saw a 24.8% drop, making used EVs the best bargain in years. Used EVs are now 8% cheaper than comparable gas cars!"
            >
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={priceDrops} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                        <XAxis type="number" stroke="#71717a" unit="%" />
                        <YAxis dataKey="model" type="category" stroke="#71717a" width={120} />
                        <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                        <Bar dataKey="drop" fill="#22c55e" radius={[0, 8, 8, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </ChartModal>

            {/* Tax Credit */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">💵 Used EV Tax Credit</h3>
                <div className="grid-3" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-green)' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-green)' }}>
                            ${data.usedEVMarket.taxCredit.amount.toLocaleString()}
                        </div>
                        <div style={{ color: 'var(--text-secondary)' }}>Max Credit</div>
                    </div>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-blue)' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-blue)' }}>
                            ${data.usedEVMarket.taxCredit.priceLimit.toLocaleString()}
                        </div>
                        <div style={{ color: 'var(--text-secondary)' }}>Price Limit</div>
                    </div>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-purple)' }}>
                        <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-purple)' }}>
                            {data.usedEVMarket.taxCredit.availability}
                        </div>
                        <div style={{ color: 'var(--text-secondary)' }}>Availability</div>
                    </div>
                </div>
            </div>

            {/* V2G Section */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">⚡ Vehicle-to-Grid (V2G) Adoption</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>2024 Market Size</span>
                                <span style={{ fontWeight: 700, color: 'var(--accent-green)' }}>${data.v2g.marketSize2024}B</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>2034 Projection</span>
                                <span style={{ fontWeight: 700, color: 'var(--accent-blue)' }}>${data.v2g.marketSize2034}B</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '1rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>CAGR (2025-2034)</span>
                                <span style={{ fontWeight: 700, color: 'var(--accent-purple)' }}>{data.v2g.cagr}%</span>
                            </div>
                        </div>
                    </div>
                    <div>
                        <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>V2G Leaders</h4>
                        {data.v2g.leaders.slice(0, 5).map((l, i) => (
                            <div key={i} className="card" style={{ padding: '0.5rem 1rem', marginBottom: '0.25rem', display: 'flex', justifyContent: 'space-between' }}>
                                <span>{l.country}</span>
                                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>{l.status}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Insurance */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🛡️ EV Insurance Costs</h3>
                <div className="grid-3" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>USA</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-red)' }}>+{data.insurance.premiumDifference.us}%</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>vs gas cars</div>
                    </div>
                    <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Australia</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-orange)' }}>+{data.insurance.premiumDifference.australia}%</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>vs gas cars</div>
                    </div>
                    <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>UK</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-green)' }}>{data.insurance.premiumDifference.uk}%</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>cheaper!</div>
                    </div>
                </div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '1rem' }}>
                    <strong>Why higher?</strong> {data.insurance.reasons.slice(0, 2).join('; ')}
                </p>
            </div>
        </div>
    )
}
