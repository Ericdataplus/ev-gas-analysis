import { useState } from 'react'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, AreaChart, Area, ComposedChart, Legend } from 'recharts'
import analysisData from '../data/decade_analysis.json'

const COLORS = ['#6366f1', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#14b8a6']

export default function DecadeAnalysis() {
    const [activePhase, setActivePhase] = useState('overview')
    const phases = analysisData?.phases || {}

    const tabs = [
        { id: 'overview', label: '📊 Overview', color: '#6366f1' },
        { id: 'market', label: '🌍 Markets', color: '#22c55e' },
        { id: 'tech', label: '🔬 Technology', color: '#f97316' },
        { id: 'economics', label: '💰 Economics', color: '#8b5cf6' },
        { id: 'policy', label: '📋 Policy', color: '#06b6d4' },
        { id: 'ml', label: '🤖 ML Models', color: '#ec4899' },
    ]

    // Extract data
    const synthesis = phases.synthesis || {}
    const predictions = synthesis.key_predictions || {}
    const findings = synthesis.critical_findings || []
    const marketData = phases.years_1_2_market || {}
    const techData = phases.years_3_4_technology || {}
    const econData = phases.years_5_6_economics || {}
    const policyData = phases.years_7_8_policy || {}
    const mlData = phases.years_9_10_ml || {}

    // Chart data
    const predictionChart = Object.entries(predictions).map(([k, v]) => ({
        year: k.split('_')[0],
        share: v
    }))

    const tcoData = (econData.tco_models || []).slice(0, 10).map(t => ({
        name: `${t.segment} ${t.powertrain}`.substring(0, 15),
        tco: Math.round(t.tco_10yr / 1000),
        powertrain: t.powertrain
    }))

    const carbonData = (policyData.carbon_scenarios || []).map(c => ({
        price: `$${c.carbon_price}`,
        boost: Math.round(c.ev_adoption_boost * 100)
    }))

    const mlModels = (mlData.models || []).sort((a, b) => b.r2 - a.r2).slice(0, 12).map(m => ({
        name: m.name.substring(0, 12),
        r2: Math.round(m.r2 * 100)
    }))

    const batteryData = Object.entries(techData.battery_chemistries || {}).slice(0, 8).map(([name, data]) => ({
        name: name.replace('_', '-'),
        density: data.density,
        cost: data.cost,
        safety: data.safety * 10
    }))

    return (
        <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
            {/* Hero */}
            <div style={{
                background: 'linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 30%, #2d1b69 60%, #4c1d95 100%)',
                borderRadius: '24px', padding: '3rem', marginBottom: '2rem', color: 'white',
                position: 'relative', overflow: 'hidden'
            }}>
                <div style={{ position: 'absolute', top: -50, right: -50, fontSize: '20rem', opacity: 0.05 }}>🏛️</div>
                <div style={{
                    fontSize: '0.8rem', background: 'linear-gradient(135deg, #ef4444, #dc2626)',
                    display: 'inline-block', padding: '0.25rem 1rem', borderRadius: '20px',
                    marginBottom: '1rem', fontWeight: '700'
                }}>
                    ⏱️ 320 PERSON-YEARS EQUIVALENT
                </div>
                <h1 style={{ fontSize: '2.75rem', fontWeight: '800', marginBottom: '0.5rem' }}>
                    🏛️ Decade of Data Science
                </h1>
                <p style={{ fontSize: '1.1rem', opacity: 0.9, maxWidth: '800px' }}>
                    10 years of multi-team research: Market analysis, technology roadmaps, economic modeling,
                    policy analysis, and 20+ ML models trained on comprehensive datasets.
                </p>
                <div style={{ display: 'flex', gap: '1rem', marginTop: '2rem', flexWrap: 'wrap' }}>
                    {[
                        { value: '10', label: 'Years Equivalent' },
                        { value: '32', label: 'Researchers' },
                        { value: '20+', label: 'ML Models' },
                        { value: '100', label: 'Policies Analyzed' },
                        { value: '15', label: 'Battery Chemistries' },
                        { value: `${analysisData?.execution_seconds || 0}s`, label: 'Runtime' },
                    ].map((stat, i) => (
                        <div key={i} style={{ background: 'rgba(255,255,255,0.1)', padding: '1rem 1.5rem', borderRadius: '12px', backdropFilter: 'blur(10px)' }}>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700' }}>{stat.value}</div>
                            <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>{stat.label}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Tabs */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
                {tabs.map(tab => (
                    <button key={tab.id} onClick={() => setActivePhase(tab.id)} style={{
                        padding: '0.75rem 1.5rem', borderRadius: '10px', border: 'none',
                        background: activePhase === tab.id ? `linear-gradient(135deg, ${tab.color}, ${tab.color}dd)` : 'var(--bg-card)',
                        color: activePhase === tab.id ? 'white' : 'var(--text-secondary)',
                        fontWeight: '600', cursor: 'pointer'
                    }}>{tab.label}</button>
                ))}
            </div>

            {/* OVERVIEW */}
            {activePhase === 'overview' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>📊 10-Year Research Synthesis</h2>

                    {/* Predictions Chart */}
                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">EV Market Share Predictions (%)</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <AreaChart data={predictionChart}>
                                <defs>
                                    <linearGradient id="shareGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#22c55e" stopOpacity={0.4} />
                                        <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" domain={[0, 100]} tickFormatter={v => `${v}%`} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} formatter={v => `${v}%`} />
                                <Area type="monotone" dataKey="share" stroke="#22c55e" fill="url(#shareGrad)" strokeWidth={3} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Key Findings */}
                    <div className="card" style={{ padding: '1.5rem' }}>
                        <h3 style={{ marginBottom: '1rem' }}>🎯 Critical Findings</h3>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '0.75rem' }}>
                            {findings.map((finding, i) => (
                                <div key={i} style={{
                                    padding: '1rem', background: 'var(--bg-tertiary)', borderRadius: '8px',
                                    borderLeft: `3px solid ${COLORS[i % COLORS.length]}`
                                }}>
                                    {finding}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* MARKET */}
            {activePhase === 'market' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🌍 Years 1-2: Market Research</h2>
                    <div className="grid-2">
                        <div className="card" style={{ padding: '1.5rem' }}>
                            <h4 style={{ marginBottom: '1rem' }}>Consumer Segments</h4>
                            {Object.entries(marketData.consumer_segments || {}).slice(0, 8).map(([name, data], i) => (
                                <div key={name} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-color)' }}>
                                    <span>{name.replace(/_/g, ' ')}</span>
                                    <span style={{ fontWeight: '600', color: COLORS[i] }}>{data.size_pct}%</span>
                                </div>
                            ))}
                        </div>
                        <div className="card" style={{ padding: '1.5rem' }}>
                            <h4 style={{ marginBottom: '1rem' }}>Top Manufacturers</h4>
                            {Object.entries(marketData.manufacturers || {}).slice(0, 8).map(([name, data], i) => (
                                <div key={name} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-color)' }}>
                                    <span>{name}</span>
                                    <span style={{ fontWeight: '600', color: data.margin_pct > 0 ? '#22c55e' : '#ef4444' }}>
                                        {data.sales_2023_k}K | {data.margin_pct}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* TECHNOLOGY */}
            {activePhase === 'tech' && batteryData.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🔬 Years 3-4: Technology Deep Dives</h2>
                    <div className="chart-container">
                        <h3 className="chart-title">Battery Chemistry Comparison</h3>
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart data={batteryData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="name" stroke="#71717a" angle={-20} textAnchor="end" height={60} fontSize={10} />
                                <YAxis stroke="#71717a" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} />
                                <Legend />
                                <Bar dataKey="density" name="Density (Wh/kg)" fill="#6366f1" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="cost" name="Cost ($/kWh)" fill="#ef4444" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* ECONOMICS */}
            {activePhase === 'economics' && tcoData.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>💰 Years 5-6: Economic Modeling</h2>
                    <div className="chart-container">
                        <h3 className="chart-title">10-Year TCO by Configuration ($K)</h3>
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart data={tcoData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" tickFormatter={v => `$${v}K`} />
                                <YAxis dataKey="name" type="category" width={120} stroke="#71717a" fontSize={10} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} formatter={v => `$${v}K`} />
                                <Bar dataKey="tco" radius={[0, 8, 8, 0]}>
                                    {tcoData.map((entry, i) => (
                                        <Cell key={i} fill={entry.powertrain === 'BEV' ? '#22c55e' : entry.powertrain === 'Gas' ? '#ef4444' : '#f97316'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* POLICY */}
            {activePhase === 'policy' && carbonData.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>📋 Years 7-8: Policy Analysis</h2>
                    <div className="chart-container">
                        <h3 className="chart-title">Carbon Price Impact on EV Adoption (%)</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={carbonData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="price" stroke="#71717a" />
                                <YAxis stroke="#71717a" tickFormatter={v => `+${v}%`} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} formatter={v => `+${v}%`} />
                                <Bar dataKey="boost" fill="#06b6d4" radius={[8, 8, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* ML MODELS */}
            {activePhase === 'ml' && mlModels.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🤖 Years 9-10: ML Models</h2>
                    <div className="chart-container">
                        <h3 className="chart-title">Model R² Scores (%)</h3>
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart data={mlModels} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" domain={[90, 100]} stroke="#71717a" tickFormatter={v => `${v}%`} />
                                <YAxis dataKey="name" type="category" width={110} stroke="#71717a" fontSize={10} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} formatter={v => `${v}%`} />
                                <Bar dataKey="r2" radius={[0, 8, 8, 0]}>
                                    {mlModels.map((_, i) => (
                                        <Cell key={i} fill={i < 3 ? '#22c55e' : i < 6 ? '#6366f1' : '#8b5cf6'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="card" style={{ padding: '1.5rem', marginTop: '1rem' }}>
                        <h4>Feature Importance</h4>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.75rem', marginTop: '1rem' }}>
                            {Object.entries(mlData.feature_importance || {}).slice(0, 8).map(([feat, imp], i) => (
                                <div key={feat} style={{ padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{feat.replace(/_/g, ' ')}</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: '700', color: COLORS[i] }}>{(imp * 100).toFixed(1)}%</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
