import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ComposedChart, Line, Legend, PieChart, Pie } from 'recharts'
import ChartModal from '../components/ChartModal'
import partData from '../data/part_complexity.json'

export default function PartComplexity() {
    const chartData = partData.chart_data
    const insights = partData.insights
    const maintenance = partData.maintenance
    const failurePoints = partData.failure_points
    const manufacturing = partData.manufacturing
    const tco = partData.ten_year_tco

    // Moving parts comparison (log scale friendly)
    const movingPartsData = [
        { type: 'Electric', parts: 20, color: '#22c55e' },
        { type: 'Hydrogen', parts: 500, color: '#3b82f6' },
        { type: 'Gasoline', parts: 2000, color: '#ef4444' },
        { type: 'Diesel', parts: 2200, color: '#78716c' },
        { type: 'Hybrid', parts: 2300, color: '#f97316' },
        { type: 'PHEV', parts: 2400, color: '#eab308' }
    ]

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🔧 Vehicle Part Complexity</h1>
                <p className="page-subtitle">Comparing mechanical complexity: Gas vs Electric vs Hybrid vs Hydrogen</p>
            </header>

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

            {/* Total Parts Comparison */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📊 Total Parts by Vehicle Type</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Total Parts Comparison"
                        insight="Electric vehicles have 50% fewer parts than gasoline vehicles. Hybrids are the most complex because they have BOTH an ICE engine AND electric components - the worst of both worlds for complexity."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData.parts_comparison}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="type" stroke="#71717a" />
                                <YAxis stroke="#71717a" tickFormatter={(v) => `${v / 1000}k`} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => v.toLocaleString()}
                                />
                                <Bar dataKey="total" name="Total Parts" radius={[8, 8, 0, 0]}>
                                    {chartData.parts_comparison.map((entry, i) => (
                                        <Cell key={i} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Parts Breakdown</h4>
                        {chartData.parts_comparison.map((item, i) => (
                            <div key={i} className="card" style={{ padding: '0.6rem', marginBottom: '0.4rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                        <span style={{
                                            width: '10px',
                                            height: '10px',
                                            borderRadius: '50%',
                                            background: item.color
                                        }}></span>
                                        {item.type}
                                    </span>
                                    <span style={{ fontWeight: 600 }}>{item.total.toLocaleString()} parts</span>
                                </div>
                            </div>
                        ))}
                        <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(34, 197, 94, 0.1)', borderRadius: '8px' }}>
                            <strong style={{ color: 'var(--accent-green)' }}>EV Advantage:</strong>
                            <p style={{ fontSize: '0.85rem', margin: '0.25rem 0 0' }}>
                                50% fewer parts = 50% fewer things that can break
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Moving Parts - The Critical Metric */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">⚙️ Moving Parts Comparison (The Real Story)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Moving Parts Count"
                        insight="This is where EVs truly shine. An electric motor has about 20 moving parts. A gasoline engine has 2,000+. Moving parts wear out, require lubrication, and eventually fail. Fewer moving parts = dramatically longer lifespan."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={movingPartsData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" />
                                <YAxis dataKey="type" type="category" stroke="#71717a" width={80} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => `${v.toLocaleString()} moving parts`}
                                />
                                <Bar dataKey="parts" radius={[0, 8, 8, 0]}>
                                    {movingPartsData.map((entry, i) => (
                                        <Cell key={i} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Why Moving Parts Matter</h4>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <strong>⚡ Electric: 20 moving parts</strong>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                Just the motor rotor, bearings, and a few pumps. No pistons, no valves, no transmission gears.
                            </p>
                        </div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.5rem', borderLeft: '3px solid #ef4444' }}>
                            <strong>⛽ Gasoline: 2,000+ moving parts</strong>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                Pistons, rods, crankshaft, camshafts, valves, timing chain, transmission gears, torque converter, pumps...
                            </p>
                        </div>
                        <div className="card" style={{ padding: '1rem', borderLeft: '3px solid #f97316' }}>
                            <strong>🔋⛽ Hybrid: 2,300+ moving parts</strong>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                All the ICE parts PLUS electric motor components. Maximum complexity.
                            </p>
                        </div>
                        <div style={{ marginTop: '1rem', padding: '1rem', background: '#27272a', borderRadius: '8px', textAlign: 'center' }}>
                            <span style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-green)' }}>100x</span>
                            <p style={{ margin: '0.25rem 0 0', fontSize: '0.9rem' }}>fewer moving parts in EVs vs gas engines</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Maintenance Comparison */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">💰 Annual Maintenance Cost</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Yearly Maintenance Expenses"
                        insight="EVs save $800/year in maintenance vs gasoline vehicles. Over 10 years, that's $8,000 in savings. No oil changes, no transmission service, no exhaust repairs, no spark plugs."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData.maintenance_cost}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="type" stroke="#71717a" />
                                <YAxis stroke="#71717a" tickFormatter={(v) => `$${v}`} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => `$${v.toLocaleString()}/year`}
                                />
                                <Bar dataKey="annual" radius={[8, 8, 0, 0]}>
                                    {chartData.maintenance_cost.map((entry, i) => (
                                        <Cell key={i} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Maintenance Items Comparison</h4>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                            <div className="card" style={{ padding: '0.75rem' }}>
                                <h5 style={{ color: '#ef4444', margin: '0 0 0.5rem' }}>⛽ Gas Maintenance</h5>
                                <ul style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', paddingLeft: '1rem', margin: 0 }}>
                                    {maintenance.gas.maintenance_items.slice(0, 6).map((item, i) => (
                                        <li key={i}>{item}</li>
                                    ))}
                                </ul>
                            </div>
                            <div className="card" style={{ padding: '0.75rem' }}>
                                <h5 style={{ color: '#22c55e', margin: '0 0 0.5rem' }}>⚡ EV Maintenance</h5>
                                <ul style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', paddingLeft: '1rem', margin: 0 }}>
                                    {maintenance.electric.maintenance_items.map((item, i) => (
                                        <li key={i}>{item}</li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                        <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: 'rgba(34, 197, 94, 0.1)', borderRadius: '8px', textAlign: 'center' }}>
                            <span style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-green)' }}>$8,000</span>
                            <p style={{ margin: '0.25rem 0 0', fontSize: '0.85rem' }}>10-year maintenance savings with EV</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Reliability Scores */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">✅ Reliability Scores</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Reliability Rating (out of 10)"
                        insight="EVs lead in reliability (8.8/10) thanks to fewer failure points. Hybrids score well (7.5) due to Toyota's engineering, but still have more potential issues than pure EVs."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData.reliability_scores} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" domain={[0, 10]} stroke="#71717a" />
                                <YAxis dataKey="type" type="category" stroke="#71717a" width={80} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => `${v}/10`}
                                />
                                <Bar dataKey="score" radius={[0, 8, 8, 0]}>
                                    {chartData.reliability_scores.map((entry, i) => (
                                        <Cell key={i} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Common Failure Points</h4>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid #ef4444' }}>
                            <strong>⛽ Gas: 8 Critical Systems</strong>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                Transmission ($4,000), Engine ($2,000), Alternator, Starter, Exhaust...
                            </p>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid #22c55e' }}>
                            <strong>⚡ Electric: 3 Critical Systems</strong>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                12V battery ($200), Onboard charger (rare), Charge port (rare)
                            </p>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', borderLeft: '3px solid #3b82f6' }}>
                            <strong>💧 Hydrogen: 5 Critical Systems</strong>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                Fuel cell stack ($15,000 if fails), Air compressor, H2 sensors
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* 10-Year TCO */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📊 10-Year Total Cost of Ownership</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="TCO Breakdown"
                        insight="Despite higher purchase prices, EVs have the lowest 10-year TCO at $93,000 vs $106,000 for gas. Hydrogen is the most expensive due to fuel costs (~$16/kg). Fuel and maintenance savings make up the EV price difference."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData.tco_breakdown}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="category" stroke="#71717a" />
                                <YAxis stroke="#71717a" tickFormatter={(v) => `$${v / 1000}k`} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => `$${v.toLocaleString()}`}
                                />
                                <Legend />
                                <Bar dataKey="Gas" fill="#ef4444" />
                                <Bar dataKey="Electric" fill="#22c55e" />
                                <Bar dataKey="Hybrid" fill="#f97316" />
                                <Bar dataKey="Hydrogen" fill="#3b82f6" />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>10-Year Total Cost</h4>
                        {Object.entries(tco).map(([type, data]) => (
                            <div key={type} className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ textTransform: 'capitalize' }}>{type}</span>
                                    <span style={{
                                        fontWeight: 700,
                                        fontSize: '1.1rem',
                                        color: type === 'electric' ? 'var(--accent-green)' :
                                            type === 'hydrogen' ? 'var(--accent-red)' : 'inherit'
                                    }}>
                                        ${data.total.toLocaleString()}
                                    </span>
                                </div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', gap: '0.5rem', marginTop: '0.25rem', flexWrap: 'wrap' }}>
                                    <span>Fuel: ${data.fuel_cost.toLocaleString()}</span>
                                    <span>•</span>
                                    <span>Maint: ${data.maintenance.toLocaleString()}</span>
                                    <span>•</span>
                                    <span>Repairs: ${data.repairs.toLocaleString()}</span>
                                </div>
                            </div>
                        ))}
                        <div style={{ marginTop: '0.75rem', padding: '1rem', background: 'rgba(34, 197, 94, 0.2)', borderRadius: '8px', textAlign: 'center' }}>
                            <p style={{ margin: 0, fontWeight: 700, color: 'var(--accent-green)' }}>
                                💰 EV saves $13,000 vs Gas over 10 years
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Manufacturing Complexity */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🏭 Manufacturing Complexity</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Assembly Time (Hours)"
                        insight="EVs require 44% less assembly time than gas vehicles (10 vs 18 hours). This translates to lower labor costs and higher manufacturing efficiency. Hybrids take the longest due to dual powertrain integration."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData.assembly_time}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="type" stroke="#71717a" />
                                <YAxis stroke="#71717a" unit=" hrs" />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => `${v} hours`}
                                />
                                <Bar dataKey="hours" radius={[8, 8, 0, 0]}>
                                    {chartData.assembly_time.map((entry, i) => (
                                        <Cell key={i} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Manufacturing Metrics</h4>
                        <table style={{ width: '100%', fontSize: '0.85rem' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                    <th style={{ textAlign: 'left', padding: '0.5rem' }}>Metric</th>
                                    <th style={{ textAlign: 'center', padding: '0.5rem' }}>Gas</th>
                                    <th style={{ textAlign: 'center', padding: '0.5rem' }}>EV</th>
                                    <th style={{ textAlign: 'center', padding: '0.5rem' }}>Hybrid</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                    <td style={{ padding: '0.5rem' }}>Assembly Time</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem' }}>18 hrs</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem', color: 'var(--accent-green)' }}>10 hrs</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem', color: 'var(--accent-red)' }}>22 hrs</td>
                                </tr>
                                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                    <td style={{ padding: '0.5rem' }}>Suppliers</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem' }}>500</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem', color: 'var(--accent-green)' }}>200</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem', color: 'var(--accent-red)' }}>600</td>
                                </tr>
                                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                    <td style={{ padding: '0.5rem' }}>Automation %</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem' }}>65%</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem', color: 'var(--accent-green)' }}>90%</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem' }}>60%</td>
                                </tr>
                                <tr>
                                    <td style={{ padding: '0.5rem' }}>Mfg Steps</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem' }}>800</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem', color: 'var(--accent-green)' }}>300</td>
                                    <td style={{ textAlign: 'center', padding: '0.5rem', color: 'var(--accent-red)' }}>1,000</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* Bottom Line */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1))' }}>
                <h3 className="chart-title">🏁 The Bottom Line</h3>
                <div style={{ marginTop: '1rem' }}>
                    <div className="grid-2">
                        <div className="card" style={{ padding: '1.5rem', borderLeft: '4px solid var(--accent-green)' }}>
                            <h4 style={{ color: 'var(--accent-green)', margin: '0 0 0.5rem' }}>⚡ Electric Vehicles Win</h4>
                            <ul style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', paddingLeft: '1.25rem', margin: 0 }}>
                                <li>50% fewer total parts</li>
                                <li><strong>100x</strong> fewer moving parts</li>
                                <li>$800/year maintenance savings</li>
                                <li>8.8/10 reliability score</li>
                                <li>$13,000 lower 10-year TCO</li>
                            </ul>
                        </div>
                        <div className="card" style={{ padding: '1.5rem', borderLeft: '4px solid var(--accent-orange)' }}>
                            <h4 style={{ color: 'var(--accent-orange)', margin: '0 0 0.5rem' }}>⚠️ Hybrids: Most Complex</h4>
                            <ul style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', paddingLeft: '1.25rem', margin: 0 }}>
                                <li>More parts than gas OR electric</li>
                                <li>Both powertrains = double maintenance</li>
                                <li>More potential failure points</li>
                                <li>Longest assembly time</li>
                                <li>Bridge technology, not destination</li>
                            </ul>
                        </div>
                    </div>
                    <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(34, 197, 94, 0.2)', borderRadius: '8px', textAlign: 'center' }}>
                        <p style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: '1.1rem', margin: 0 }}>
                            🔧 Simplicity is the ultimate sophistication. EVs prove that less is more.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
