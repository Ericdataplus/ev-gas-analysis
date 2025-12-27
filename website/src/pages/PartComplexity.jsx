import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ComposedChart, Line, Legend, PieChart, Pie, LineChart, Area } from 'recharts'
import ChartModal from '../components/ChartModal'
import partData from '../data/part_complexity.json'
import deepDiveData from '../data/part_complexity_deep_dive.json'

export default function PartComplexity() {
    const [activeTab, setActiveTab] = useState('overview')

    // Part Complexity Data
    const chartData = partData.chart_data
    const insights = partData.insights
    const maintenance = partData.maintenance
    const failurePoints = partData.failure_points
    const tco = partData.ten_year_tco
    const componentPricing = partData.component_pricing
    const logistics = partData.logistics

    // Deep Dive Data
    const timeEvolution = deepDiveData.time_evolution
    const repairAnalysis = deepDiveData.repair_analysis
    const manufacturing = deepDiveData.manufacturing_innovations
    const geopolitical = deepDiveData.geopolitical_risks
    const lifecycle = deepDiveData.lifecycle
    const components = deepDiveData.component_deep_dives
    const reliability = deepDiveData.reliability_data
    const business = deepDiveData.business_impact
    const futureTech = deepDiveData.future_tech
    const partsBreakdown = deepDiveData.detailed_parts_breakdown
    const charts = deepDiveData.charts
    const deepInsights = deepDiveData.key_insights

    // Moving parts comparison
    const movingPartsData = [
        { type: 'Electric', parts: 20, color: '#22c55e' },
        { type: 'Hydrogen', parts: 500, color: '#3b82f6' },
        { type: 'Gasoline', parts: 2000, color: '#ef4444' },
        { type: 'Diesel', parts: 2200, color: '#78716c' },
        { type: 'Hybrid', parts: 2300, color: '#f97316' },
        { type: 'PHEV', parts: 2400, color: '#eab308' }
    ]

    const tabs = [
        { id: 'overview', label: '📊 Overview' },
        { id: 'costs', label: '💰 Parts Costs' },
        { id: 'battery', label: '🔋 Battery Tech' },
        { id: 'manufacturing', label: '🏭 Manufacturing' },
        { id: 'repair', label: '🔧 Repair' },
        { id: 'future', label: '🚀 Future Tech' }
    ]

    return (
        <div>
            {/* Header */}
            <div style={{
                background: 'linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #3b82f6 100%)',
                borderRadius: '20px',
                padding: '2rem',
                marginBottom: '2rem',
                color: 'white'
            }}>
                <h1 style={{ fontSize: '2rem', fontWeight: '800', marginBottom: '0.5rem' }}>
                    🔧 Vehicle Parts Analysis
                </h1>
                <p style={{ opacity: 0.9, maxWidth: '600px' }}>
                    Comprehensive breakdown: part counts, costs, complexity, repairability, and future technology impact
                </p>
                <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>20</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>EV Moving Parts</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>2,000</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Gas Moving Parts</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>90%</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Battery Cost Drop Since 2010</div>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div style={{
                display: 'flex',
                gap: '0.5rem',
                marginBottom: '2rem',
                overflowX: 'auto',
                paddingBottom: '0.5rem'
            }}>
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        style={{
                            padding: '0.75rem 1.25rem',
                            borderRadius: '10px',
                            border: 'none',
                            background: activeTab === tab.id
                                ? 'linear-gradient(135deg, #2563eb, #1d4ed8)'
                                : 'var(--bg-card)',
                            color: activeTab === tab.id ? 'white' : 'var(--text-secondary)',
                            fontWeight: '600',
                            cursor: 'pointer',
                            whiteSpace: 'nowrap',
                            transition: 'all 0.2s'
                        }}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Overview Tab */}
            {activeTab === 'overview' && (
                <div>
                    {/* Key Stats */}
                    <div className="stats-grid">
                        <div className="stat-card" style={{ borderTop: '3px solid #ef4444' }}>
                            <div className="stat-icon">⛽</div>
                            <div className="stat-value">30,000</div>
                            <div className="stat-label">Gas Engine Parts</div>
                            <div className="stat-change">2,000 moving parts</div>
                        </div>
                        <div className="stat-card" style={{ borderTop: '3px solid #22c55e' }}>
                            <div className="stat-icon">⚡</div>
                            <div className="stat-value">15,000</div>
                            <div className="stat-label">Electric Vehicle Parts</div>
                            <div className="stat-change" style={{ color: 'var(--accent-green)' }}>Only 20 moving parts!</div>
                        </div>
                        <div className="stat-card" style={{ borderTop: '3px solid #f97316' }}>
                            <div className="stat-icon">🔋⛽</div>
                            <div className="stat-value">35,000</div>
                            <div className="stat-label">Hybrid Parts</div>
                            <div className="stat-change">Most complex = most parts</div>
                        </div>
                        <div className="stat-card" style={{ borderTop: '3px solid #3b82f6' }}>
                            <div className="stat-icon">💧</div>
                            <div className="stat-value">22,000</div>
                            <div className="stat-label">Hydrogen Fuel Cell Parts</div>
                            <div className="stat-change">500 moving parts</div>
                        </div>
                    </div>

                    {/* Key Insights */}
                    <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                        <h3 className="chart-title">💡 Key Findings</h3>
                        <div className="grid-2" style={{ marginTop: '1rem' }}>
                            {insights.slice(0, 6).map((insight, i) => (
                                <div key={i} className="card" style={{
                                    padding: '1rem',
                                    borderLeft: `3px solid ${insight.category === 'parts' ? 'var(--accent-green)' :
                                        insight.category === 'maintenance' ? 'var(--accent-blue)' :
                                            insight.category === 'hybrid' ? 'var(--accent-orange)' :
                                                insight.category === 'hydrogen' ? '#3b82f6' :
                                                    'var(--accent-purple)'
                                        }`
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                        <span style={{ fontSize: '1.4rem' }}>{insight.icon}</span>
                                        <strong>{insight.title}</strong>
                                    </div>
                                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                                        {insight.detail}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Moving Parts Chart */}
                    <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                        <h3 className="chart-title">⚙️ Moving Parts Comparison</h3>
                        <ChartModal
                            title="Moving Parts by Vehicle Type"
                            insight="Electric vehicles have 100x fewer moving parts than gas vehicles. This is why EVs require less maintenance - fewer parts means fewer things that can break."
                        >
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={movingPartsData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                    <XAxis dataKey="type" stroke="#71717a" />
                                    <YAxis stroke="#71717a" scale="log" domain={[10, 3000]} />
                                    <Tooltip
                                        contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    />
                                    <Bar dataKey="parts" radius={[8, 8, 0, 0]}>
                                        {movingPartsData.map((entry, i) => (
                                            <Cell key={i} fill={entry.color} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </ChartModal>
                    </div>
                </div>
            )}

            {/* Parts Costs Tab */}
            {activeTab === 'costs' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>💰 Detailed Parts Cost Breakdown</h2>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', marginTop: '1rem' }}>
                        {/* Gas Vehicle Parts */}
                        <div className="card" style={{ padding: '1rem' }}>
                            <h4 style={{ color: '#ef4444', marginBottom: '0.75rem' }}>⛽ Gasoline - ${partsBreakdown.gas.total_parts_cost.toLocaleString()} parts cost</h4>
                            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                                <table style={{ width: '100%', fontSize: '0.75rem' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                                            <th style={{ textAlign: 'left', padding: '0.25rem' }}>#</th>
                                            <th style={{ textAlign: 'left', padding: '0.25rem' }}>Part</th>
                                            <th style={{ textAlign: 'right', padding: '0.25rem' }}>Cost</th>
                                            <th style={{ textAlign: 'right', padding: '0.25rem' }}>%</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {partsBreakdown.gas.top_20_parts.map((part, i) => (
                                            <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                                <td style={{ padding: '0.25rem', color: 'var(--text-secondary)' }}>{part.rank}</td>
                                                <td style={{ padding: '0.25rem' }}>{part.part}</td>
                                                <td style={{ textAlign: 'right', padding: '0.25rem', fontWeight: i < 3 ? 600 : 400 }}>${part.cost.toLocaleString()}</td>
                                                <td style={{ textAlign: 'right', padding: '0.25rem', color: 'var(--text-secondary)' }}>{part.percent}%</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <div style={{ marginTop: '0.75rem', padding: '0.5rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '4px', fontSize: '0.75rem' }}>
                                💡 {partsBreakdown.gas.expensive_part_insight}
                            </div>
                        </div>

                        {/* Electric Vehicle Parts */}
                        <div className="card" style={{ padding: '1rem' }}>
                            <h4 style={{ color: '#22c55e', marginBottom: '0.75rem' }}>⚡ Electric - ${partsBreakdown.electric.total_parts_cost.toLocaleString()} parts cost</h4>
                            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                                <table style={{ width: '100%', fontSize: '0.75rem' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                                            <th style={{ textAlign: 'left', padding: '0.25rem' }}>#</th>
                                            <th style={{ textAlign: 'left', padding: '0.25rem' }}>Part</th>
                                            <th style={{ textAlign: 'right', padding: '0.25rem' }}>Cost</th>
                                            <th style={{ textAlign: 'right', padding: '0.25rem' }}>%</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {partsBreakdown.electric.top_20_parts.map((part, i) => (
                                            <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                                <td style={{ padding: '0.25rem', color: 'var(--text-secondary)' }}>{part.rank}</td>
                                                <td style={{ padding: '0.25rem' }}>{part.part}</td>
                                                <td style={{ textAlign: 'right', padding: '0.25rem', fontWeight: i < 3 ? 600 : 400 }}>${part.cost.toLocaleString()}</td>
                                                <td style={{ textAlign: 'right', padding: '0.25rem', color: 'var(--text-secondary)' }}>{part.percent}%</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <div style={{ marginTop: '0.75rem', padding: '0.5rem', background: 'rgba(34, 197, 94, 0.1)', borderRadius: '4px', fontSize: '0.75rem' }}>
                                💡 {partsBreakdown.electric.expensive_part_insight}
                            </div>
                        </div>

                        {/* Hybrid Vehicle Parts */}
                        <div className="card" style={{ padding: '1rem' }}>
                            <h4 style={{ color: '#f97316', marginBottom: '0.75rem' }}>🔋⛽ Hybrid - ${partsBreakdown.hybrid.total_parts_cost.toLocaleString()} parts cost</h4>
                            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                                <table style={{ width: '100%', fontSize: '0.75rem' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                                            <th style={{ textAlign: 'left', padding: '0.25rem' }}>#</th>
                                            <th style={{ textAlign: 'left', padding: '0.25rem' }}>Part</th>
                                            <th style={{ textAlign: 'right', padding: '0.25rem' }}>Cost</th>
                                            <th style={{ textAlign: 'right', padding: '0.25rem' }}>%</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {partsBreakdown.hybrid.top_20_parts.map((part, i) => (
                                            <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                                <td style={{ padding: '0.25rem', color: 'var(--text-secondary)' }}>{part.rank}</td>
                                                <td style={{ padding: '0.25rem' }}>{part.part}</td>
                                                <td style={{ textAlign: 'right', padding: '0.25rem', fontWeight: i < 3 ? 600 : 400 }}>${part.cost.toLocaleString()}</td>
                                                <td style={{ textAlign: 'right', padding: '0.25rem', color: 'var(--text-secondary)' }}>{part.percent}%</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <div style={{ marginTop: '0.75rem', padding: '0.5rem', background: 'rgba(249, 115, 22, 0.1)', borderRadius: '4px', fontSize: '0.75rem' }}>
                                💡 {partsBreakdown.hybrid.expensive_part_insight}
                            </div>
                        </div>

                        {/* Hydrogen Vehicle Parts */}
                        <div className="card" style={{ padding: '1rem' }}>
                            <h4 style={{ color: '#3b82f6', marginBottom: '0.75rem' }}>💧 Hydrogen - ${partsBreakdown.hydrogen.total_parts_cost.toLocaleString()} parts cost</h4>
                            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                                <table style={{ width: '100%', fontSize: '0.75rem' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                                            <th style={{ textAlign: 'left', padding: '0.25rem' }}>#</th>
                                            <th style={{ textAlign: 'left', padding: '0.25rem' }}>Part</th>
                                            <th style={{ textAlign: 'right', padding: '0.25rem' }}>Cost</th>
                                            <th style={{ textAlign: 'right', padding: '0.25rem' }}>%</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {partsBreakdown.hydrogen.top_20_parts.map((part, i) => (
                                            <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                                <td style={{ padding: '0.25rem', color: 'var(--text-secondary)' }}>{part.rank}</td>
                                                <td style={{ padding: '0.25rem' }}>{part.part}</td>
                                                <td style={{ textAlign: 'right', padding: '0.25rem', fontWeight: i < 3 ? 600 : 400 }}>${part.cost.toLocaleString()}</td>
                                                <td style={{ textAlign: 'right', padding: '0.25rem', color: 'var(--text-secondary)' }}>{part.percent}%</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <div style={{ marginTop: '0.75rem', padding: '0.5rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '4px', fontSize: '0.75rem' }}>
                                💡 {partsBreakdown.hydrogen.expensive_part_insight}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Battery Tech Tab */}
            {activeTab === 'battery' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🔋 Battery Technology Deep Dive</h2>

                    {/* Battery Cost Evolution */}
                    <div className="chart-container">
                        <h3 className="chart-title">📉 Battery Cost Revolution (2010-2030)</h3>
                        <div className="grid-2" style={{ marginTop: '1rem' }}>
                            <ChartModal
                                title="Battery Pack Cost per kWh"
                                insight="Battery costs have dropped 90% since 2010. At $115/kWh in 2024, a 75kWh pack costs $8,625. By 2030 at $58/kWh, that same pack will cost $4,350."
                            >
                                <ResponsiveContainer width="100%" height="100%">
                                    <ComposedChart data={charts.battery_cost_timeline}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                        <XAxis dataKey="year" stroke="#71717a" />
                                        <YAxis stroke="#71717a" tickFormatter={(v) => `$${v}`} />
                                        <Tooltip
                                            contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                            formatter={(v) => `$${v}/kWh`}
                                        />
                                        <Area type="monotone" dataKey="cost" fill="rgba(34, 197, 94, 0.2)" stroke="#22c55e" strokeWidth={2} />
                                        <Line type="monotone" dataKey="cost" stroke="#22c55e" strokeWidth={2} dot={{ fill: '#22c55e', r: 3 }} />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </ChartModal>

                            <div>
                                <h4 style={{ marginBottom: '0.75rem' }}>Cost Parity Timeline</h4>
                                <div className="card" style={{ padding: '1rem', marginBottom: '0.5rem', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1))' }}>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                                        <div>
                                            <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-green)' }}>90%</div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Cost drop since 2010</div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-blue)' }}>2026</div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Manufacturing parity</div>
                                        </div>
                                    </div>
                                </div>
                                <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                                    <strong>2024:</strong> 75kWh pack = <span style={{ color: 'var(--accent-orange)' }}>${(115 * 75).toLocaleString()}</span>
                                </div>
                                <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                                    <strong>2030:</strong> 75kWh pack = <span style={{ color: 'var(--accent-green)' }}>${(58 * 75).toLocaleString()}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Battery Chemistry Comparison */}
                    <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                        <h3 className="chart-title">⚗️ Battery Chemistry Comparison</h3>
                        <div style={{ marginTop: '1rem', overflowX: 'auto' }}>
                            <table style={{ width: '100%', fontSize: '0.85rem' }}>
                                <thead>
                                    <tr style={{ borderBottom: '2px solid var(--border-color)' }}>
                                        <th style={{ textAlign: 'left', padding: '0.75rem' }}>Chemistry</th>
                                        <th style={{ textAlign: 'center', padding: '0.75rem' }}>Energy Density</th>
                                        <th style={{ textAlign: 'center', padding: '0.75rem' }}>Cycle Life</th>
                                        <th style={{ textAlign: 'center', padding: '0.75rem' }}>Cost $/kWh</th>
                                        <th style={{ textAlign: 'center', padding: '0.75rem' }}>Cobalt</th>
                                        <th style={{ textAlign: 'left', padding: '0.75rem' }}>Used By</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.entries(components.battery_chemistry).map(([key, chem], i) => (
                                        <tr key={key} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                            <td style={{ padding: '0.75rem', fontWeight: 600 }}>{chem.name}</td>
                                            <td style={{ textAlign: 'center', padding: '0.75rem' }}>{chem.energy_density_wh_kg} Wh/kg</td>
                                            <td style={{ textAlign: 'center', padding: '0.75rem' }}>{chem.cycle_life?.toLocaleString()}</td>
                                            <td style={{ textAlign: 'center', padding: '0.75rem', color: chem.cost_per_kwh < 100 ? 'var(--accent-green)' : 'inherit' }}>${chem.cost_per_kwh}</td>
                                            <td style={{ textAlign: 'center', padding: '0.75rem', color: chem.cobalt_content_percent === 0 ? 'var(--accent-green)' : 'inherit' }}>
                                                {chem.cobalt_content_percent !== undefined ? `${chem.cobalt_content_percent}%` : 'TBD'}
                                            </td>
                                            <td style={{ padding: '0.75rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                                                {chem.used_by?.slice(0, 2).join(', ') || chem.leaders?.slice(0, 2).join(', ')}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {/* Manufacturing Tab */}
            {activeTab === 'manufacturing' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🏭 Manufacturing Innovations</h2>

                    {/* Gigacasting */}
                    <div className="chart-container">
                        <h3 className="chart-title">🏭 Gigacasting Revolution</h3>
                        <div className="grid-2" style={{ marginTop: '1rem' }}>
                            <div>
                                <h4 style={{ marginBottom: '0.75rem' }}>What is Gigacasting?</h4>
                                <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.9rem' }}>
                                    {manufacturing.gigacasting.definition}
                                </p>
                                <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                                    <h5 style={{ marginBottom: '0.5rem' }}>Impact Numbers</h5>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem' }}>
                                        <div>
                                            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-green)' }}>70-100 → 2</div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Parts eliminated</div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-green)' }}>700 → 50</div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Welds eliminated</div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-green)' }}>40%</div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Cost reduction</div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-green)' }}>30%</div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Faster assembly</div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div>
                                <h4 style={{ marginBottom: '0.75rem' }}>Adoption Timeline</h4>
                                {manufacturing.gigacasting.adopters_timeline.map((item, i) => (
                                    <div key={i} className="card" style={{ padding: '0.6rem', marginBottom: '0.4rem', borderLeft: `3px solid ${i < 2 ? 'var(--accent-green)' : 'var(--accent-blue)'}` }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <strong>{item.company}</strong>
                                            <span style={{ color: 'var(--text-secondary)' }}>{item.year}</span>
                                        </div>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{item.model}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Manufacturing Cost Parity */}
                    <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                        <h3 className="chart-title">⚖️ Manufacturing Cost Parity</h3>
                        <ChartModal
                            title="When EVs Become Cheaper to Build"
                            insight="EV manufacturing costs are falling while ICE costs rise due to emissions compliance. Crossover happens in 2026."
                        >
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={charts.manufacturing_parity}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                    <XAxis dataKey="year" stroke="#71717a" />
                                    <YAxis stroke="#71717a" tickFormatter={(v) => `$${v / 1000}k`} domain={[22000, 38000]} />
                                    <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                    <Legend />
                                    <Line type="monotone" dataKey="ice" name="ICE" stroke="#ef4444" strokeWidth={2} />
                                    <Line type="monotone" dataKey="ev" name="EV" stroke="#22c55e" strokeWidth={2} />
                                    <Line type="monotone" dataKey="hybrid" name="Hybrid" stroke="#f97316" strokeWidth={2} />
                                </LineChart>
                            </ResponsiveContainer>
                        </ChartModal>
                    </div>
                </div>
            )}

            {/* Repair Tab */}
            {activeTab === 'repair' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🔧 Repairability & Right to Repair</h2>

                    <div className="grid-2">
                        <ChartModal
                            title="DIY Repairability Score"
                            insight="Gas vehicles are easiest to repair yourself (7.5/10). EVs are challenging (4.0/10) due to high-voltage safety."
                        >
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={charts.repairability_scores} layout="vertical">
                                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                    <XAxis type="number" domain={[0, 10]} stroke="#71717a" />
                                    <YAxis dataKey="type" type="category" stroke="#71717a" width={80} />
                                    <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                    <Bar dataKey="diy" name="DIY Score" radius={[0, 8, 8, 0]}>
                                        {charts.repairability_scores.map((entry, i) => (
                                            <Cell key={i} fill={entry.color} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </ChartModal>

                        <div>
                            <h4 style={{ marginBottom: '0.75rem' }}>Independent Mechanic Capability</h4>
                            {charts.repairability_scores.map((item, i) => (
                                <div key={i} className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: item.color }}></span>
                                            {item.type}
                                        </span>
                                        <span style={{ fontWeight: 600 }}>{item.indie}% can service</span>
                                    </div>
                                    <div style={{ height: '6px', background: '#27272a', borderRadius: '3px', overflow: 'hidden' }}>
                                        <div style={{ height: '100%', width: `${item.indie}%`, background: item.color, borderRadius: '3px' }}></div>
                                    </div>
                                </div>
                            ))}
                            <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px', fontSize: '0.85rem' }}>
                                ⚠️ Only 35% of mechanics can service EVs
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Future Tech Tab */}
            {activeTab === 'future' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🚀 Future Technology Impact</h2>

                    <div className="grid-2">
                        <div className="card" style={{ padding: '1rem' }}>
                            <h4 style={{ color: 'var(--accent-purple)', marginBottom: '0.75rem' }}>⚡ Solid-State Batteries (2027-2030)</h4>
                            <ul style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', paddingLeft: '1rem' }}>
                                <li><strong>2x energy density</strong> (400 Wh/kg vs 200)</li>
                                <li><strong>10-min charging</strong> (10-80%)</li>
                                <li><strong>Near-zero fire risk</strong></li>
                                <li><strong>40% fewer thermal parts</strong></li>
                            </ul>
                            <div style={{ marginTop: '0.75rem', fontSize: '0.75rem' }}>
                                Leaders: {futureTech.solid_state_batteries.leaders.join(', ')}
                            </div>
                        </div>

                        <div className="card" style={{ padding: '1rem' }}>
                            <h4 style={{ color: 'var(--accent-blue)', marginBottom: '0.75rem' }}>🔌 Vehicle-to-Grid</h4>
                            <ul style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', paddingLeft: '1rem' }}>
                                <li>Car becomes backup power source</li>
                                <li>${futureTech.vehicle_to_grid.potential_value.grid_services_annual}/year grid services revenue</li>
                                <li>${futureTech.vehicle_to_grid.potential_value.home_backup_value} home backup value</li>
                            </ul>
                            <div style={{ marginTop: '0.75rem', fontSize: '0.75rem' }}>
                                Available now: {futureTech.vehicle_to_grid.current_vehicles.join(', ')}
                            </div>
                        </div>
                    </div>

                    {/* Summary */}
                    <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(59, 130, 246, 0.15))' }}>
                        <h3 className="chart-title">🏁 Summary</h3>
                        <div className="grid-2" style={{ marginTop: '1rem' }}>
                            <div className="card" style={{ padding: '1.5rem', borderLeft: '4px solid var(--accent-green)' }}>
                                <h4 style={{ color: 'var(--accent-green)', marginBottom: '0.75rem' }}>✅ EV Advantages</h4>
                                <ul style={{ fontSize: '0.85rem', margin: 0, paddingLeft: '1rem' }}>
                                    <li>Battery costs down 90% since 2010</li>
                                    <li>Manufacturing parity by 2026</li>
                                    <li>90% recyclability with second-life value</li>
                                    <li>40% lower lifetime CO2</li>
                                    <li>67% lower service costs</li>
                                </ul>
                            </div>
                            <div className="card" style={{ padding: '1.5rem', borderLeft: '4px solid var(--accent-orange)' }}>
                                <h4 style={{ color: 'var(--accent-orange)', marginBottom: '0.75rem' }}>⚠️ Challenges</h4>
                                <ul style={{ fontSize: '0.85rem', margin: 0, paddingLeft: '1rem' }}>
                                    <li>Supply chain concentration (China)</li>
                                    <li>Only 35% of mechanics EV-capable</li>
                                    <li>Right to repair limited</li>
                                    <li>Charging infrastructure gaps</li>
                                    <li>Legacy OEMs losing money on EVs</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
