import { AreaChart, Area, LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ComposedChart, Legend } from 'recharts'
import ChartModal from '../components/ChartModal'
import batteryData from '../data/battery_predictions.json'

export default function BatteryAnalysis() {
    // Combine Li-Ion historical + predictions
    const liIonCombined = [
        ...batteryData.densityTrajectory.liIon.historical,
        ...batteryData.densityTrajectory.liIon.predictions.map(p => ({ ...p, predicted: true }))
    ]

    // Cost trajectory
    const costCombined = [
        ...batteryData.costTrajectory.historical,
        ...batteryData.costTrajectory.predictions.map(p => ({ ...p, predicted: true }))
    ]

    // Li-Ion vs Solid-State comparison
    const densityComparison = [
        { year: 2024, liIon: 350, solidState: null },
        { year: 2025, liIon: 370, solidState: 400 },
        { year: 2027, liIon: 400, solidState: 500 },
        { year: 2030, liIon: 450, solidState: 600 },
        { year: 2035, liIon: 480, solidState: 800 },
        { year: 2040, liIon: 500, solidState: 900 },
    ]

    // EV Range projections
    const rangeData = [
        { year: 2020, range: 200 },
        { year: 2024, range: 260 },
        { year: 2027, range: 350 },
        { year: 2030, range: 450 },
        { year: 2035, range: 600 },
    ]

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🔋 Battery Technology Deep Dive</h1>
                <p className="page-subtitle">ML-powered analysis: Cost, density, and solid-state trajectories</p>
            </header>

            {/* Key Metrics */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon">📉</div>
                    <div className="stat-value">90%</div>
                    <div className="stat-label">Cost Drop Since 2010</div>
                    <div className="stat-change">$1,100 → $115/kWh</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">📈</div>
                    <div className="stat-value">4.4x</div>
                    <div className="stat-label">Density Improvement</div>
                    <div className="stat-change">80 → 350 Wh/kg</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">⚡</div>
                    <div className="stat-value">2027</div>
                    <div className="stat-label">Solid-State Mass Production</div>
                    <div className="stat-change">Toyota, BYD, CATL</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🚗</div>
                    <div className="stat-value">600 mi</div>
                    <div className="stat-label">EV Range by 2035</div>
                    <div className="stat-change">With 800 Wh/kg batteries</div>
                </div>
            </div>

            {/* Cost Trajectory */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">💰 Battery Cost Trajectory ($/kWh)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Historical + ML Predicted Costs"
                        insight={`Battery costs dropped from $1,100/kWh in 2010 to $115/kWh in 2024 - a 90% reduction! Following Wright's Law (${batteryData.costTrajectory.annualDeclineRate}% annual decline), costs should reach ~$65/kWh by 2030 and ~$45/kWh by 2040. This is what makes EVs inevitably cheaper than ICE.`}
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={costCombined}>
                                <defs>
                                    <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" unit="$" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} formatter={(v) => `$${v}/kWh`} />
                                <Area type="monotone" dataKey="cost" stroke="#22c55e" fill="url(#costGrad)" strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Wright's Law Predictions</h4>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Annual Decline Rate</span>
                                <span style={{ fontWeight: 600, color: 'var(--accent-green)' }}>{batteryData.costTrajectory.annualDeclineRate}%</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-blue)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Costs halve every</span>
                                <span style={{ fontWeight: 600, color: 'var(--accent-blue)' }}>{batteryData.costTrajectory.halfingPeriodYears} years</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>2030 Projection</span>
                                <span style={{ fontWeight: 600 }}>$65/kWh</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '0.75rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>2040 Projection</span>
                                <span style={{ fontWeight: 600 }}>$45/kWh</span>
                            </div>
                        </div>
                        <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'var(--bg-hover)', borderRadius: '8px' }}>
                            <p style={{ fontSize: '0.85rem', margin: 0, color: 'var(--text-muted)' }}>
                                💡 EVs reach cost parity with ICE at ~$80/kWh (by 2028)
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Energy Density - Li-Ion vs Solid-State */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">⚡ Energy Density: Li-Ion vs Solid-State (Wh/kg)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Li-Ion vs Solid-State Trajectory"
                        insight="Solid-state batteries will SURPASS Li-ion by 2027. Li-ion is approaching its theoretical limit (~500 Wh/kg), but solid-state can reach 800-1000 Wh/kg. This enables 600+ mile EVs and 10-minute charging!"
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={densityComparison}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" domain={[300, 1000]} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Legend />
                                <Line type="monotone" dataKey="liIon" stroke="#3b82f6" strokeWidth={2} name="Li-Ion" dot={{ r: 4 }} />
                                <Line type="monotone" dataKey="solidState" stroke="#a855f7" strokeWidth={3} name="Solid-State" dot={{ r: 5 }} strokeDasharray="5 5" />
                            </LineChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Theoretical Limits</h4>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.75rem', borderLeft: '3px solid var(--accent-blue)' }}>
                            <h5 style={{ color: 'var(--accent-blue)', margin: 0 }}>Li-Ion: ~500 Wh/kg</h5>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                Approaching limit. NMC/NCA cathodes maxed out.
                            </p>
                        </div>
                        <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-purple)' }}>
                            <h5 style={{ color: 'var(--accent-purple)', margin: 0 }}>Solid-State: ~1000 Wh/kg</h5>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                Lithium-metal anodes + solid electrolyte unlock 2x capacity.
                            </p>
                        </div>
                        <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(168, 85, 247, 0.2)', borderRadius: '8px' }}>
                            <p style={{ fontSize: '0.9rem', margin: 0, fontWeight: 600, color: 'var(--accent-purple)' }}>
                                🚀 Solid-state overtakes Li-ion by 2027!
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* EV Range Projections */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🚗 EV Range Evolution (miles)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Average EV Range Over Time"
                        insight="EV range has grown from 73 miles (2010 Nissan Leaf) to 260+ miles today. By 2030, 450-mile ranges will be standard. By 2035, 600-mile EVs with solid-state batteries will be common."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={rangeData}>
                                <defs>
                                    <linearGradient id="rangeGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" domain={[0, 700]} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} formatter={(v) => `${v} miles`} />
                                <Area type="monotone" dataKey="range" stroke="#f97316" fill="url(#rangeGrad)" strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>What Enables This?</h4>
                        {Object.entries(batteryData.evRangeProjections).map(([year, data]) => (
                            <div key={year} className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                    <span style={{ fontWeight: 600 }}>{year}</span>
                                    <span style={{ color: 'var(--accent-orange)', fontWeight: 600 }}>{data.range} miles</span>
                                </div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                    {data.packKwh} kWh · {data.density} Wh/kg · {data.tech}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Chemistry Comparison */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🧪 Battery Chemistry Comparison</h3>
                <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                                <th style={{ textAlign: 'left', padding: '0.75rem', color: 'var(--text-muted)' }}>Chemistry</th>
                                <th style={{ textAlign: 'center', padding: '0.75rem', color: 'var(--text-muted)' }}>Wh/kg</th>
                                <th style={{ textAlign: 'center', padding: '0.75rem', color: 'var(--text-muted)' }}>Cycles</th>
                                <th style={{ textAlign: 'center', padding: '0.75rem', color: 'var(--text-muted)' }}>Cost</th>
                                <th style={{ textAlign: 'center', padding: '0.75rem', color: 'var(--text-muted)' }}>Safety</th>
                            </tr>
                        </thead>
                        <tbody>
                            {batteryData.chemistryComparison.map((chem, i) => (
                                <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                    <td style={{ padding: '0.75rem', fontWeight: 500 }}>{chem.type}</td>
                                    <td style={{ padding: '0.75rem', textAlign: 'center' }}>
                                        <span style={{
                                            color: chem.whKg >= 500 ? 'var(--accent-green)' : chem.whKg >= 300 ? 'var(--accent-blue)' : 'var(--text-primary)'
                                        }}>{chem.whKg}</span>
                                    </td>
                                    <td style={{ padding: '0.75rem', textAlign: 'center' }}>{chem.cycles.toLocaleString()}</td>
                                    <td style={{ padding: '0.75rem', textAlign: 'center' }}>{chem.cost}</td>
                                    <td style={{ padding: '0.75rem', textAlign: 'center' }}>
                                        <span style={{
                                            color: chem.safety === 'Excellent' ? 'var(--accent-green)' : 'var(--accent-blue)'
                                        }}>{chem.safety}</span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Key Milestones */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📅 Battery Technology Milestones</h3>
                <div style={{ marginTop: '1rem' }}>
                    {batteryData.keyMilestones.map((m, i) => (
                        <div key={i} style={{
                            display: 'flex',
                            gap: '1rem',
                            padding: '0.75rem 0',
                            borderLeft: `3px solid ${m.year <= 2024 ? 'var(--accent-green)' : 'var(--accent-purple)'}`,
                            paddingLeft: '1rem',
                            marginBottom: '0.5rem'
                        }}>
                            <span style={{
                                fontWeight: 700,
                                color: m.year <= 2024 ? 'var(--accent-green)' : 'var(--accent-purple)',
                                minWidth: '50px'
                            }}>{m.year}</span>
                            <span style={{ color: 'var(--text-secondary)' }}>{m.event}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Bottom Line */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(168, 85, 247, 0.1))' }}>
                <h3 className="chart-title">🎯 Bottom Line: Battery Technology is Exponential</h3>
                <div style={{ marginTop: '1rem', fontSize: '1.05rem', lineHeight: 1.8 }}>
                    <p><strong style={{ color: 'var(--accent-green)' }}>📉 COST:</strong> From $1,100 → $115/kWh (90% drop). Heading to $45/kWh by 2040.</p>
                    <p><strong style={{ color: 'var(--accent-blue)' }}>📈 DENSITY:</strong> Li-ion approaching 500 Wh/kg limit. Game over for ICE.</p>
                    <p><strong style={{ color: 'var(--accent-purple)' }}>⚡ SOLID-STATE:</strong> 800+ Wh/kg by 2035. 600-mile EVs. 10-min charging.</p>
                    <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(34, 197, 94, 0.2)', borderRadius: '8px', textAlign: 'center' }}>
                        <p style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: '1.1rem', margin: 0 }}>
                            By 2030, battery EVs will be unambiguously cheaper, better, and more convenient than gas cars.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
