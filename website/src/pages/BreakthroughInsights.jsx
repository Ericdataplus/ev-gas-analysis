import { useState, useEffect } from 'react'
import {
    AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
    ComposedChart, ReferenceLine, RadarChart, PolarGrid, PolarAngleAxis,
    PolarRadiusAxis, Radar
} from 'recharts'

// Load breakthrough analysis data
import analysisData from '../data/breakthrough_analysis.json'

const COLORS = {
    critical: '#ef4444',
    warning: '#f97316',
    ok: '#22c55e',
    primary: '#6366f1',
    secondary: '#8b5cf6',
    info: '#06b6d4'
}

export default function BreakthroughInsights() {
    const [activeTab, setActiveTab] = useState('grid')

    const analyses = analysisData?.analyses || {}
    const gridData = analyses.grid_failure || {}
    const supplyData = analyses.supply_chain_cascade || {}
    const climateData = analyses.climate_ev_paradox || {}
    const insuranceData = analyses.insurance_spiral || {}
    const wealthData = analyses.wealth_multiplier || {}
    const chargingData = analyses.charging_deserts || {}
    const neuralData = analyses.gpu_neural_network || {}

    const tabs = [
        { id: 'grid', label: '⚡ Grid Failure', icon: '⚡' },
        { id: 'supply', label: '🔗 Supply Chain', icon: '🔗' },
        { id: 'climate', label: '🌡️ Climate Paradox', icon: '🌡️' },
        { id: 'insurance', label: '📉 Insurance Spiral', icon: '📉' },
        { id: 'wealth', label: '💰 Wealth Gap', icon: '💰' },
        { id: 'charging', label: '🏜️ Charging Deserts', icon: '🏜️' },
        { id: 'neural', label: '🧠 AI Predictions', icon: '🧠' }
    ]

    const getStatusColor = (status) => {
        if (status === 'FAILURE' || status === 'STOPPED') return COLORS.critical
        if (status === 'CRITICAL' || status === 'FAILING') return COLORS.warning
        return COLORS.ok
    }

    return (
        <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
            {/* Header */}
            <div style={{
                background: 'linear-gradient(135deg, #1e1b4b 0%, #4c1d95 50%, #7c3aed 100%)',
                borderRadius: '20px',
                padding: '2.5rem',
                marginBottom: '2rem',
                color: 'white',
                position: 'relative',
                overflow: 'hidden'
            }}>
                <div style={{ position: 'absolute', top: -20, right: -20, fontSize: '12rem', opacity: 0.1 }}>🧠</div>
                <h1 style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>
                    🔮 Breakthrough Insights
                </h1>
                <p style={{ fontSize: '1.1rem', opacity: 0.9, maxWidth: '700px' }}>
                    Questions nobody has asked — answered with GPU-accelerated machine learning
                    and deep statistical analysis. Novel findings that challenge assumptions.
                </p>
                <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>7</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Novel Analyses</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>RTX 3060</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>GPU Accelerated</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>5,000</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Training Samples</div>
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
                                ? 'linear-gradient(135deg, #7c3aed, #5b21b6)'
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

            {/* GRID FAILURE TAB */}
            {activeTab === 'grid' && gridData.scenarios && (
                <div>
                    <div style={{ background: '#fef2f2', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem', borderLeft: '4px solid #ef4444' }}>
                        <h2 style={{ fontSize: '1.5rem', fontWeight: '700', color: '#991b1b', marginBottom: '0.5rem' }}>
                            ⚡ At what EV adoption % does the US grid FAIL?
                        </h2>
                        <p style={{ color: '#7f1d1d' }}>
                            Nobody has modeled this precisely. We simulated EV charging load vs grid capacity.
                        </p>
                    </div>

                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">Grid Stress vs EV Adoption</h3>
                        <ResponsiveContainer width="100%" height={400}>
                            <ComposedChart data={gridData.scenarios}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="adoption_pct" tickFormatter={v => `${v}%`} stroke="#71717a" />
                                <YAxis stroke="#71717a" domain={[0, 150]} tickFormatter={v => `${v}%`} />
                                <Tooltip
                                    formatter={(v, name) => [name === 'grid_stress_pct' ? `${v}%` : v, name]}
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a' }}
                                />
                                <Legend />
                                <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="5 5" label="FAILURE" />
                                <ReferenceLine y={85} stroke="#f97316" strokeDasharray="3 3" label="Stressed" />
                                <Area type="monotone" dataKey="grid_stress_pct" name="Grid Stress %" fill="#ef444440" stroke="#ef4444" />
                                <Bar dataKey="peak_charging_gw" name="Peak Charging (GW)" fill="#6366f1" />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                        {gridData.insights?.map((insight, i) => (
                            <div key={i} className="card" style={{
                                padding: '1rem',
                                borderLeft: `3px solid ${i < 2 ? '#ef4444' : '#f97316'}`
                            }}>
                                {insight}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* SUPPLY CHAIN TAB */}
            {activeTab === 'supply' && supplyData.cascade_scenarios && (
                <div>
                    <div style={{ background: '#fef3c7', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem', borderLeft: '4px solid #f59e0b' }}>
                        <h2 style={{ fontSize: '1.5rem', fontWeight: '700', color: '#92400e', marginBottom: '0.5rem' }}>
                            🔗 If Taiwan is blocked, how many days until car production STOPS?
                        </h2>
                        <p style={{ color: '#78350f' }}>
                            We modeled inventory buffers and cascade failures across critical supply nodes.
                        </p>
                    </div>

                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">Days Until Production Shutdown (By Supply Node)</h3>
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart data={supplyData.cascade_scenarios} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" />
                                <YAxis dataKey="node" type="category" width={150} stroke="#71717a" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} />
                                <Bar dataKey="days_to_shutdown" name="Days to Shutdown" fill="#f97316" radius={[0, 8, 8, 0]}>
                                    {supplyData.cascade_scenarios.map((entry, i) => (
                                        <Cell key={i} fill={entry.days_to_shutdown < 50 ? '#ef4444' : '#f97316'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    {supplyData.taiwan_blockade && (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
                            <div className="card" style={{ padding: '1.25rem', borderTop: '3px solid #ef4444' }}>
                                <div style={{ fontSize: '2rem', fontWeight: '700', color: '#ef4444' }}>{supplyData.taiwan_blockade.days_until_production_stops}</div>
                                <div>Days until auto stops</div>
                            </div>
                            <div className="card" style={{ padding: '1.25rem', borderTop: '3px solid #f97316' }}>
                                <div style={{ fontSize: '2rem', fontWeight: '700', color: '#f97316' }}>{supplyData.taiwan_blockade.us_vehicles_per_day_lost?.toLocaleString()}</div>
                                <div>Vehicles/day lost</div>
                            </div>
                            <div className="card" style={{ padding: '1.25rem', borderTop: '3px solid #8b5cf6' }}>
                                <div style={{ fontSize: '2rem', fontWeight: '700', color: '#8b5cf6' }}>${supplyData.taiwan_blockade.economic_impact_per_day_b}B</div>
                                <div>Daily economic loss</div>
                            </div>
                            <div className="card" style={{ padding: '1.25rem', borderTop: '3px solid #06b6d4' }}>
                                <div style={{ fontSize: '2rem', fontWeight: '700', color: '#06b6d4' }}>{supplyData.taiwan_blockade.recovery_time_years}+ years</div>
                                <div>Recovery time</div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* CLIMATE PARADOX TAB */}
            {activeTab === 'climate' && climateData.city_analysis && (
                <div>
                    <div style={{ background: '#fef2f2', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem', borderLeft: '4px solid #ef4444' }}>
                        <h2 style={{ fontSize: '1.5rem', fontWeight: '700', color: '#991b1b', marginBottom: '0.5rem' }}>
                            🌡️ THE PARADOX: Climate change hurts EVs in hot regions
                        </h2>
                        <p style={{ color: '#7f1d1d' }}>
                            Cities that need EVs most (hot, polluted) are where EV batteries perform WORST.
                        </p>
                    </div>

                    <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
                        <div className="chart-container">
                            <h3 className="chart-title">Range Loss by City (2024 → 2050)</h3>
                            <ResponsiveContainer width="100%" height={350}>
                                <BarChart data={climateData.city_analysis.slice(0, 8)}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                    <XAxis dataKey="city" stroke="#71717a" />
                                    <YAxis stroke="#71717a" />
                                    <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} />
                                    <Bar dataKey="range_loss_miles" name="Miles Lost" fill="#ef4444" radius={[8, 8, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="chart-container">
                            <h3 className="chart-title">Battery Efficiency vs Temperature</h3>
                            <ResponsiveContainer width="100%" height={350}>
                                <AreaChart data={climateData.temp_efficiency_curve}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                    <XAxis dataKey="temp" tickFormatter={v => `${v}°C`} stroke="#71717a" />
                                    <YAxis domain={[50, 100]} tickFormatter={v => `${v}%`} stroke="#71717a" />
                                    <Tooltip formatter={v => `${v}%`} contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} />
                                    <Area type="monotone" dataKey="efficiency" fill="#22c55e40" stroke="#22c55e" strokeWidth={2} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {climateData.paradox && (
                        <div className="card" style={{ padding: '1.5rem', background: 'linear-gradient(135deg, rgba(239,68,68,0.1), rgba(249,115,22,0.1))' }}>
                            <h4 style={{ marginBottom: '0.75rem', color: '#dc2626' }}>🔥 The Finding</h4>
                            <p style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>{climateData.paradox.finding}</p>
                            <p style={{ color: 'var(--text-secondary)' }}>
                                {climateData.paradox.population_affected_millions}M Americans live in heat-affected regions
                            </p>
                        </div>
                    )}
                </div>
            )}

            {/* INSURANCE SPIRAL TAB */}
            {activeTab === 'insurance' && insuranceData.vehicle_analysis && (
                <div>
                    <div style={{ background: '#fef2f2', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem', borderLeft: '4px solid #ef4444' }}>
                        <h2 style={{ fontSize: '1.5rem', fontWeight: '700', color: '#991b1b', marginBottom: '0.5rem' }}>
                            📉 The Insurance Death Spiral: When EVs become uninsurable
                        </h2>
                        <p style={{ color: '#7f1d1d' }}>
                            Battery replacement cost eventually exceeds car value — making insurance irrational.
                        </p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                        {insuranceData.vehicle_analysis.slice(0, 4).map((car, i) => (
                            <div key={i} className="card" style={{ padding: '1.25rem' }}>
                                <h4 style={{ marginBottom: '0.75rem' }}>{car.model}</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.9rem' }}>
                                    <div>MSRP: <strong>${car.msrp.toLocaleString()}</strong></div>
                                    <div>Battery: <strong>${car.battery_cost.toLocaleString()}</strong></div>
                                </div>
                                <div style={{
                                    marginTop: '0.75rem',
                                    padding: '0.5rem',
                                    background: car.years_until_uninsurable < 8 ? '#fee2e2' : '#fef3c7',
                                    borderRadius: '8px',
                                    textAlign: 'center',
                                    fontWeight: '600',
                                    color: car.years_until_uninsurable < 8 ? '#dc2626' : '#d97706'
                                }}>
                                    Uninsurable at Year {car.years_until_uninsurable}
                                </div>
                            </div>
                        ))}
                    </div>

                    {insuranceData.summary && (
                        <div className="chart-container">
                            <h3 className="chart-title">Key Finding</h3>
                            <div style={{ fontSize: '1.25rem', padding: '1rem' }}>
                                Average EV becomes economically uninsurable at <strong style={{ color: '#ef4444' }}>year {insuranceData.summary.avg_years_to_uninsurable}</strong>.
                                This creates a structural problem for the used EV market.
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* WEALTH GAP TAB */}
            {activeTab === 'wealth' && wealthData.yearly_comparison && (
                <div>
                    <div style={{ background: '#f0fdf4', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem', borderLeft: '4px solid #22c55e' }}>
                        <h2 style={{ fontSize: '1.5rem', fontWeight: '700', color: '#166534', marginBottom: '0.5rem' }}>
                            💰 The 20-Year Wealth Multiplier: EV + Solar + Home vs Traditional
                        </h2>
                        <p style={{ color: '#15803d' }}>
                            Compound effect of sustainable choices vs renting + gas car.
                        </p>
                    </div>

                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">Wealth Accumulation Over 20 Years</h3>
                        <ResponsiveContainer width="100%" height={400}>
                            <AreaChart data={wealthData.yearly_comparison}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis tickFormatter={v => `$${v / 1000}k`} stroke="#71717a" />
                                <Tooltip formatter={v => `$${v.toLocaleString()}`} contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} />
                                <Legend />
                                <Area type="monotone" dataKey="sustainable_wealth" name="Sustainable Path" fill="#22c55e40" stroke="#22c55e" strokeWidth={2} />
                                <Area type="monotone" dataKey="traditional_wealth" name="Traditional Path" fill="#ef444440" stroke="#ef4444" strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>

                    {wealthData.summary && (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                            <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #22c55e', textAlign: 'center' }}>
                                <div style={{ fontSize: '2.5rem', fontWeight: '700', color: '#22c55e' }}>
                                    ${(wealthData.summary.wealth_gap_20yr / 1000).toFixed(0)}k
                                </div>
                                <div>20-Year Wealth Gap</div>
                            </div>
                            <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #6366f1', textAlign: 'center' }}>
                                <div style={{ fontSize: '2.5rem', fontWeight: '700', color: '#6366f1' }}>
                                    {wealthData.summary.wealth_multiplier}x
                                </div>
                                <div>Wealth Multiplier</div>
                            </div>
                            <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #ef4444', textAlign: 'center' }}>
                                <div style={{ fontSize: '2.5rem', fontWeight: '700', color: '#ef4444' }}>
                                    ${(wealthData.summary.cumulative_rent_lost / 1000).toFixed(0)}k
                                </div>
                                <div>Rent Lost to Landlord</div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* CHARGING DESERTS TAB */}
            {activeTab === 'charging' && chargingData.states_data && (
                <div>
                    <div style={{ background: '#fef3c7', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem', borderLeft: '4px solid #f59e0b' }}>
                        <h2 style={{ fontSize: '1.5rem', fontWeight: '700', color: '#92400e', marginBottom: '0.5rem' }}>
                            🏜️ Charging Deserts: Who's Left Behind?
                        </h2>
                        <p style={{ color: '#78350f' }}>
                            {chargingData.summary?.population_excluded_millions}M Americans live in states where EV adoption is effectively impossible.
                        </p>
                    </div>

                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">Chargers per 100k Population vs EV Adoption</h3>
                        <ResponsiveContainer width="100%" height={400}>
                            <ComposedChart data={chargingData.states_data}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="state" stroke="#71717a" angle={-45} textAnchor="end" height={80} />
                                <YAxis yAxisId="left" stroke="#71717a" />
                                <YAxis yAxisId="right" orientation="right" stroke="#71717a" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} />
                                <Legend />
                                <Bar yAxisId="left" dataKey="chargers_per_100k" name="Chargers/100k" fill="#6366f1">
                                    {chargingData.states_data.map((entry, i) => (
                                        <Cell key={i} fill={entry.chargers_per_100k < 10 ? '#ef4444' : '#6366f1'} />
                                    ))}
                                </Bar>
                                <Line yAxisId="right" type="monotone" dataKey="ev_share_pct" name="EV Share %" stroke="#22c55e" strokeWidth={2} />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="card" style={{ padding: '1rem', background: '#fee2e2' }}>
                        <h4 style={{ color: '#dc2626', marginBottom: '0.5rem' }}>🔴 Charging Desert States</h4>
                        <p>{chargingData.charging_deserts?.join(', ')}</p>
                        <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: '#7f1d1d' }}>
                            Correlation: Infrastructure ↔ Adoption: r = {chargingData.correlations?.chargers_vs_ev_adoption}
                        </p>
                    </div>
                </div>
            )}

            {/* NEURAL NETWORK TAB */}
            {activeTab === 'neural' && neuralData.model_performance && (
                <div>
                    <div style={{ background: 'linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.1))', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem', borderLeft: '4px solid #6366f1' }}>
                        <h2 style={{ fontSize: '1.5rem', fontWeight: '700', color: '#4338ca', marginBottom: '0.5rem' }}>
                            🧠 GPU Neural Network Predictions
                        </h2>
                        <p style={{ color: '#5b21b6' }}>
                            Deep learning model trained on RTX 3060 to predict EV adoption based on battery costs, gas prices, and infrastructure.
                        </p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                        <div className="card" style={{ padding: '1.25rem', borderTop: '3px solid #6366f1' }}>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#6366f1' }}>
                                {(neuralData.model_performance.r2_score * 100).toFixed(1)}%
                            </div>
                            <div>R² Score</div>
                        </div>
                        <div className="card" style={{ padding: '1.25rem', borderTop: '3px solid #8b5cf6' }}>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#8b5cf6' }}>
                                {neuralData.model_performance.epochs_trained}
                            </div>
                            <div>Epochs Trained</div>
                        </div>
                        <div className="card" style={{ padding: '1.25rem', borderTop: '3px solid #06b6d4' }}>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#06b6d4' }}>
                                {neuralData.model_performance.samples.toLocaleString()}
                            </div>
                            <div>Training Samples</div>
                        </div>
                        <div className="card" style={{ padding: '1.25rem', borderTop: '3px solid #22c55e' }}>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#22c55e' }}>
                                RTX 3060
                            </div>
                            <div>GPU Used</div>
                        </div>
                    </div>

                    {neuralData.future_predictions && (
                        <div className="chart-container">
                            <h3 className="chart-title">AI Predictions: EV Adoption Scenarios</h3>
                            <div style={{ overflowX: 'auto' }}>
                                <table style={{ width: '100%', fontSize: '0.9rem' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '2px solid var(--border-color)' }}>
                                            <th style={{ padding: '0.75rem', textAlign: 'left' }}>Year</th>
                                            <th style={{ padding: '0.75rem', textAlign: 'center' }}>Battery $/kWh</th>
                                            <th style={{ padding: '0.75rem', textAlign: 'center' }}>Gas Price</th>
                                            <th style={{ padding: '0.75rem', textAlign: 'center' }}>Predicted Adoption</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {neuralData.future_predictions.filter((p, i) => i % 2 === 0).slice(0, 12).map((pred, i) => (
                                            <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                                <td style={{ padding: '0.75rem' }}>{pred.year}</td>
                                                <td style={{ padding: '0.75rem', textAlign: 'center' }}>${pred.battery_cost}</td>
                                                <td style={{ padding: '0.75rem', textAlign: 'center' }}>${pred.gas_price}</td>
                                                <td style={{ padding: '0.75rem', textAlign: 'center', fontWeight: '600', color: pred.predicted_ev_adoption > 30 ? '#22c55e' : '#f59e0b' }}>
                                                    {pred.predicted_ev_adoption}%
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
