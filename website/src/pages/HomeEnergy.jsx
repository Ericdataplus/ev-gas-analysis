import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts'
import ChartModal from '../components/ChartModal'
import data from '../data/insights.json'

export default function HomeEnergy() {
    const heatingData = [
        { name: 'Natural Gas', value: data.homeEnergy.heatingFuel2024.naturalGas, color: '#f97316' },
        { name: 'Electric', value: data.homeEnergy.heatingFuel2024.electric, color: '#22c55e' },
        { name: 'Propane', value: data.homeEnergy.heatingFuel2024.propane, color: '#a855f7' },
        { name: 'Oil', value: data.homeEnergy.heatingFuel2024.oil, color: '#ef4444' },
        { name: 'Other', value: data.homeEnergy.heatingFuel2024.other, color: '#6b7280' },
    ]

    const electricityData = [
        { name: 'Heating/Cooling', value: data.homeEnergy.electricityBreakdown.heatingCooling, color: '#ef4444' },
        { name: 'Water Heating', value: data.homeEnergy.electricityBreakdown.waterHeating, color: '#f97316' },
        { name: 'Appliances', value: data.homeEnergy.electricityBreakdown.appliances, color: '#3b82f6' },
        { name: 'Lighting', value: data.homeEnergy.electricityBreakdown.lighting, color: '#eab308' },
        { name: 'Refrigeration', value: data.homeEnergy.electricityBreakdown.refrigeration, color: '#22c55e' },
    ]

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🏠 Home Energy</h1>
                <p className="page-subtitle">US residential energy consumption: gas vs electric breakdown</p>
            </header>

            {/* Key Stats */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon">⚡</div>
                    <div className="stat-value">{(data.homeEnergy.annualConsumption.totalKwh / 1000).toFixed(1)}K</div>
                    <div className="stat-label">Avg kWh/Year</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">❄️</div>
                    <div className="stat-value">${data.homeEnergy.annualConsumption.coolingCost2024}</div>
                    <div className="stat-label">Summer Cooling Cost</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">📈</div>
                    <div className="stat-value">+{data.homeEnergy.annualConsumption.coolingIncrease}%</div>
                    <div className="stat-label">Cooling Cost Increase</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🔥</div>
                    <div className="stat-value">{data.homeEnergy.heatingFuel2024.naturalGas}%</div>
                    <div className="stat-label">Use Gas Heating</div>
                </div>
            </div>

            <div className="grid-2">
                <ChartModal
                    title="🔥 Home Heating Fuel Mix (2024)"
                    insight="Gas heating still dominates at 47%, but electric heating has grown from 38% in 2010 to 42% today. This shift is driven by heat pumps, which are 3-4x more efficient than traditional electric resistance heating."
                >
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie data={heatingData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name}: ${value}%`}>
                                {heatingData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                            </Pie>
                            <Tooltip />
                        </PieChart>
                    </ResponsiveContainer>
                </ChartModal>

                <ChartModal
                    title="⚡ Electricity Use Breakdown"
                    insight="Heating and cooling dominate at 51% of home electricity. Water heating is second at 19%. A solar + battery system can offset most of this, especially combined with a heat pump for HVAC."
                >
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={electricityData} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis type="number" stroke="#71717a" unit="%" />
                            <YAxis dataKey="name" type="category" stroke="#71717a" width={100} />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                            <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                                {electricityData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </ChartModal>
            </div>

            {/* Trends */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📊 Energy Trends</h3>
                <div className="grid-3" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-green)' }}>
                        <h4 style={{ fontSize: '0.9rem' }}>⚡ Electric Heating Growth</h4>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.25rem' }}>
                            {data.homeEnergy.trends.electricHeatingGrowth}
                        </p>
                    </div>
                    <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-orange)' }}>
                        <h4 style={{ fontSize: '0.9rem' }}>🔥 Gas Heating Decline</h4>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.25rem' }}>
                            {data.homeEnergy.trends.gasHeatingDecline}
                        </p>
                    </div>
                    <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-red)' }}>
                        <h4 style={{ fontSize: '0.9rem' }}>🌡️ 2024 Summer</h4>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.25rem' }}>
                            {data.homeEnergy.trends.recordCooling2024}
                        </p>
                    </div>
                </div>
            </div>

            {/* Connection to EVs */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🔌 The Solar + EV Connection</h3>
                <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', lineHeight: 1.6 }}>
                    <strong>Key Insight:</strong> An average home uses {(data.homeEnergy.annualConsumption.totalKwh / 1000).toFixed(0)}K kWh/year.
                    An EV driven 12,000 miles/year needs ~3,400 kWh (29% of home usage).
                    An 8kW solar system produces ~12,000 kWh/year - enough for <strong>both home and EV</strong>.
                </p>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-green)' }}>$1,284</div>
                        <div style={{ color: 'var(--text-secondary)' }}>Annual EV fuel savings with solar</div>
                    </div>
                    <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-blue)' }}>7 years</div>
                        <div style={{ color: 'var(--text-secondary)' }}>Payback with solar + EV combo</div>
                    </div>
                </div>
            </div>
        </div>
    )
}
