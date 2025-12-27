import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, Legend, ComposedChart, Area } from 'recharts'
import ChartModal from '../components/ChartModal'
import deepDiveData from '../data/part_complexity_deep_dive.json'

export default function PartComplexityDeepDive() {
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
    const insights = deepDiveData.key_insights

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🔬 Part Complexity Deep Dive</h1>
                <p className="page-subtitle">Comprehensive analysis: Battery evolution, Manufacturing, Supply Chain, Lifecycle, and Future Tech</p>
            </header>

            {/* Key Insights Grid */}
            <div className="stats-grid">
                {insights.slice(0, 4).map((insight, i) => (
                    <div key={i} className="stat-card">
                        <div className="stat-icon">{insight.icon}</div>
                        <div className="stat-value" style={{ fontSize: '1rem' }}>{insight.title}</div>
                        <div className="stat-label" style={{ fontSize: '0.75rem' }}>{insight.detail}</div>
                    </div>
                ))}
            </div>

            {/* Battery Cost Evolution - The Key Story */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📉 Battery Cost Revolution (2010-2030)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Battery Pack Cost per kWh"
                        insight="Battery costs have dropped 90% since 2010. At $115/kWh in 2024, a 75kWh pack costs $8,625. By 2030 at $58/kWh, that same pack will cost $4,350 - making EVs cheaper to manufacture than ICE."
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
                        <div style={{ padding: '0.75rem', background: '#27272a', borderRadius: '8px', textAlign: 'center' }}>
                            <span style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-green)' }}>$4,275 savings</span>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>per pack by 2030</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Manufacturing Cost Parity */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">⚖️ Manufacturing Cost Parity Timeline</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="When EVs Become Cheaper to Build"
                        insight="EV manufacturing costs are falling while ICE costs rise due to emissions compliance. The crossover happens in 2026. By 2030, EVs will be $6,000 cheaper to manufacture."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={charts.manufacturing_parity}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" tickFormatter={(v) => `$${v / 1000}k`} domain={[22000, 38000]} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => `$${v.toLocaleString()}`}
                                />
                                <Legend />
                                <Line type="monotone" dataKey="ice" name="ICE" stroke="#ef4444" strokeWidth={2} dot={{ r: 4 }} />
                                <Line type="monotone" dataKey="ev" name="EV" stroke="#22c55e" strokeWidth={2} dot={{ r: 4 }} />
                                <Line type="monotone" dataKey="hybrid" name="Hybrid" stroke="#f97316" strokeWidth={2} dot={{ r: 4 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Key Milestones</h4>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-orange)' }}>
                            <strong>2024:</strong> EV costs $2,500 more than ICE
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <strong>2026:</strong> EV reaches cost parity with ICE
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <strong>2028:</strong> EV is $3,500 cheaper than ICE
                        </div>
                        <div className="card" style={{ padding: '0.75rem', borderLeft: '3px solid var(--accent-red)' }}>
                            <strong>Hybrid:</strong> Costs keep rising (complexity)
                        </div>
                    </div>
                </div>
            </div>

            {/* DETAILED PARTS COST BREAKDOWN - User's request */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(59, 130, 246, 0.1))' }}>
                <h3 className="chart-title">💰 Detailed Parts Cost Breakdown - Top 20 Most Expensive Parts</h3>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                    What makes up the $23k-$34k parts cost for each vehicle type?
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', marginTop: '1rem' }}>
                    {/* Gas Vehicle Parts */}
                    <div className="card" style={{ padding: '1rem' }}>
                        <h4 style={{ color: '#ef4444', marginBottom: '0.75rem' }}>⛽ Gasoline - ${partsBreakdown.gas.total_parts_cost.toLocaleString()} parts cost</h4>
                        <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                            <table style={{ width: '100%', fontSize: '0.75rem' }}>
                                <thead>
                                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                        <th style={{ textAlign: 'left', padding: '0.25rem' }}>#</th>
                                        <th style={{ textAlign: 'left', padding: '0.25rem' }}>Part</th>
                                        <th style={{ textAlign: 'right', padding: '0.25rem' }}>Cost</th>
                                        <th style={{ textAlign: 'right', padding: '0.25rem' }}>%</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {partsBreakdown.gas.top_20_parts.map((part, i) => (
                                        <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
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
                                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                        <th style={{ textAlign: 'left', padding: '0.25rem' }}>#</th>
                                        <th style={{ textAlign: 'left', padding: '0.25rem' }}>Part</th>
                                        <th style={{ textAlign: 'right', padding: '0.25rem' }}>Cost</th>
                                        <th style={{ textAlign: 'right', padding: '0.25rem' }}>%</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {partsBreakdown.electric.top_20_parts.map((part, i) => (
                                        <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
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
                                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                        <th style={{ textAlign: 'left', padding: '0.25rem' }}>#</th>
                                        <th style={{ textAlign: 'left', padding: '0.25rem' }}>Part</th>
                                        <th style={{ textAlign: 'right', padding: '0.25rem' }}>Cost</th>
                                        <th style={{ textAlign: 'right', padding: '0.25rem' }}>%</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {partsBreakdown.hybrid.top_20_parts.map((part, i) => (
                                        <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
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
                                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                        <th style={{ textAlign: 'left', padding: '0.25rem' }}>#</th>
                                        <th style={{ textAlign: 'left', padding: '0.25rem' }}>Part</th>
                                        <th style={{ textAlign: 'right', padding: '0.25rem' }}>Cost</th>
                                        <th style={{ textAlign: 'right', padding: '0.25rem' }}>%</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {partsBreakdown.hydrogen.top_20_parts.map((part, i) => (
                                        <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
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

                {/* Most Expensive Single Component Comparison */}
                <div style={{ marginTop: '1.5rem' }}>
                    <h4 style={{ marginBottom: '0.75rem' }}>🏆 Most Expensive Single Component by Vehicle Type</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                        {deepDiveData.parts_comparison_chart.most_expensive_component.map((item, i) => (
                            <div key={i} className="card" style={{ padding: '1rem', textAlign: 'center', borderTop: `3px solid ${item.color}` }}>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>${item.cost.toLocaleString()}</div>
                                <div style={{ fontWeight: 600, marginTop: '0.25rem' }}>{item.component}</div>
                                <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>{item.type}</div>
                                <div style={{ color: item.color, fontSize: '0.85rem', marginTop: '0.25rem' }}>{item.percent}% of total cost</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Repairability Analysis */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🔧 Repairability & Right to Repair</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="DIY Repairability Score"
                        insight="Gas vehicles are easiest to repair yourself (7.5/10). EVs are challenging (4.0/10) due to high-voltage safety. Hydrogen is nearly impossible (2.0/10) - requires H2 certified technicians."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={charts.repairability_scores} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" domain={[0, 10]} stroke="#71717a" />
                                <YAxis dataKey="type" type="category" stroke="#71717a" width={80} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                />
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
                            ⚠️ Only 35% of mechanics can service EVs - creates bottleneck
                        </div>
                    </div>
                </div>
            </div>

            {/* Gigacasting Revolution */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
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

            {/* Geopolitical Supply Chain Risks */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🌍 Supply Chain Geopolitical Risk</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Supply Chain Risk Score"
                        insight="EVs have higher supply chain risk (7.5/10) due to critical mineral concentration in China (rare earths), DRC (cobalt). Gas vehicles have more diversified, mature supply chains."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={charts.supply_chain_risk}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="type" stroke="#71717a" />
                                <YAxis stroke="#71717a" domain={[0, 10]} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar dataKey="risk" name="Risk Score" radius={[8, 8, 0, 0]}>
                                    {charts.supply_chain_risk.map((entry, i) => (
                                        <Cell key={i} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Critical Material Concentration</h4>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-red)' }}>
                            <strong>Rare Earths:</strong> China 90% processing
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-red)' }}>
                            <strong>Cobalt:</strong> DRC 70% production
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-orange)' }}>
                            <strong>Graphite:</strong> China 65% production
                        </div>
                        <div className="card" style={{ padding: '0.75rem', borderLeft: '3px solid var(--accent-orange)' }}>
                            <strong>Platinum:</strong> South Africa 70% (H2 fuel cells)
                        </div>
                        <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: 'rgba(34, 197, 94, 0.1)', borderRadius: '8px', fontSize: '0.85rem' }}>
                            ✅ Mitigation: LFP batteries (no cobalt), induction motors (no rare earths)
                        </div>
                    </div>
                </div>
            </div>

            {/* Battery Chemistry Comparison */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🔋 Battery Chemistry Deep Dive</h3>
                <div style={{ marginTop: '1rem' }}>
                    <table style={{ width: '100%', fontSize: '0.85rem' }}>
                        <thead>
                            <tr style={{ borderBottom: '2px solid var(--border)' }}>
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
                                <tr key={key} style={{ borderBottom: '1px solid var(--border)' }}>
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
                    <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(34, 197, 94, 0.1)', borderRadius: '8px' }}>
                        <strong>💡 LFP is the future for standard range:</strong> No cobalt, 3000 cycles, $90/kWh.
                        BYD Blade and Tesla SR already use it. Sodium-ion next at $70/kWh with no lithium!
                    </div>
                </div>
            </div>

            {/* Carbon Footprint Lifecycle */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">♻️ Lifecycle Carbon Footprint</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Total CO2 Emissions (Manufacturing + 150k miles)"
                        insight="EVs produce 40 tons CO2 over lifetime vs 65 tons for gas. With a clean grid, EVs drop to just 20 tons. The manufacturing premium (12 vs 8 tons) is quickly offset by cleaner operation."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={charts.carbon_footprint}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="type" stroke="#71717a" />
                                <YAxis stroke="#71717a" unit=" tons" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Legend />
                                <Bar dataKey="manufacturing" name="Manufacturing" stackId="a" fill="#f97316" />
                                <Bar dataKey="lifetime" name="Lifetime Use" stackId="a" fill="#3b82f6" />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Recyclability & Second Life</h4>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <strong>EV Battery Second Life:</strong> 10+ years in grid storage
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Value capture: $1,500 per pack</div>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <strong>EV Recyclability:</strong> 90% (including battery materials)
                        </div>
                        <div className="card" style={{ padding: '0.75rem', borderLeft: '3px solid var(--accent-orange)' }}>
                            <strong>ICE Recyclability:</strong> 85% (steel, aluminum)
                        </div>
                        <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                            <div className="card" style={{ padding: '0.75rem', textAlign: 'center' }}>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-green)' }}>8.5</div>
                                <div style={{ fontSize: '0.75rem' }}>EV Circular Economy Score</div>
                            </div>
                            <div className="card" style={{ padding: '0.75rem', textAlign: 'center' }}>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-orange)' }}>6.0</div>
                                <div style={{ fontSize: '0.75rem' }}>ICE Circular Economy Score</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Business Model Impact */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">💼 Business Model Disruption</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="OEM Profit Per Vehicle"
                        insight="Tesla profits $9,000 per vehicle through vertical integration and manufacturing efficiency. Most legacy automakers lose money on EVs (-$3,000) due to inefficient manufacturing and battery costs."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={charts.oem_margins}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="company" stroke="#71717a" />
                                <YAxis stroke="#71717a" tickFormatter={(v) => `$${v / 1000}k`} domain={[-5000, 10000]} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => `$${v.toLocaleString()}`}
                                />
                                <Bar dataKey="margin" name="Profit/Vehicle" radius={[8, 8, 0, 0]}>
                                    {charts.oem_margins.map((entry, i) => (
                                        <Cell key={i} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Dealer Service Revenue Impact</h4>
                        <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div>
                                    <div style={{ color: '#ef4444' }}>⛽ ICE Service Revenue</div>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>${business.dealer_service_revenue.ice_annual_per_vehicle}/year</div>
                                </div>
                                <div style={{ fontSize: '2rem' }}>→</div>
                                <div>
                                    <div style={{ color: '#22c55e' }}>⚡ EV Service Revenue</div>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>${business.dealer_service_revenue.ev_annual_per_vehicle}/year</div>
                                </div>
                            </div>
                            <div style={{ textAlign: 'center', marginTop: '0.75rem', padding: '0.5rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '4px' }}>
                                <span style={{ color: 'var(--accent-red)', fontWeight: 700 }}>{business.dealer_service_revenue.revenue_drop_percent}% revenue drop</span>
                            </div>
                        </div>
                        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                            Dealers earn 45% margin on service - EVs destroy this profit center.
                            This is why some dealers resist EV sales.
                        </p>
                    </div>
                </div>
            </div>

            {/* Future Technology */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(59, 130, 246, 0.1))' }}>
                <h3 className="chart-title">🚀 Future Technology Impact</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
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
            </div>

            {/* Bottom Summary */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(59, 130, 246, 0.15))' }}>
                <h3 className="chart-title">🏁 Deep Dive Summary</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1.5rem', borderLeft: '4px solid var(--accent-green)' }}>
                        <h4 style={{ color: 'var(--accent-green)', marginBottom: '0.75rem' }}>✅ EV Advantages Confirmed</h4>
                        <ul style={{ fontSize: '0.85rem', margin: 0, paddingLeft: '1rem' }}>
                            <li>Battery costs down 90% since 2010</li>
                            <li>Manufacturing parity by 2026</li>
                            <li>90% recyclability with second-life value</li>
                            <li>40% lower lifetime CO2</li>
                            <li>67% lower service costs</li>
                        </ul>
                    </div>
                    <div className="card" style={{ padding: '1.5rem', borderLeft: '4px solid var(--accent-orange)' }}>
                        <h4 style={{ color: 'var(--accent-orange)', marginBottom: '0.75rem' }}>⚠️ Challenges Remain</h4>
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
    )
}
