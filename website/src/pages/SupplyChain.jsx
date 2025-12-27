import { useState } from 'react'
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
    PieChart, Pie, AreaChart, Area, ComposedChart, Line, Legend
} from 'recharts'
import data from '../data/insights.json'
import ultraDeepData from '../data/ultra_deep_ml_analysis.json'
import aiSupplyData from '../data/ai_supply_chain_analysis.json'

const COLORS = {
    critical: '#ef4444',
    high: '#f97316',
    moderate: '#eab308',
    low: '#22c55e',
    primary: '#6366f1',
    secondary: '#8b5cf6',
    info: '#06b6d4'
}

export default function SupplyChain() {
    const [activeTab, setActiveTab] = useState('overview')

    // Material risk data from ultra deep analysis
    const materialRisks = ultraDeepData?.supply_chain_risk?.material_risk_scores || {}
    const evSupplyRisk = ultraDeepData?.supply_chain_risk?.ev_supply_chain_risk || {}

    // Format for charts
    const riskChartData = Object.entries(materialRisks)
        .map(([name, data]) => ({
            name,
            risk: data.composite_risk_score,
            concentration: Math.round(data.concentration_risk * 100),
            geopolitical: Math.round(data.geopolitical_risk * 100),
            volatility: Math.round(data.price_volatility * 100),
            demandGrowth: Math.round(data.demand_growth * 100),
            reserveYears: data.reserve_years,
            level: data.risk_level
        }))
        .sort((a, b) => b.risk - a.risk)

    // Radar data for risk breakdown
    const radarData = [
        { subject: 'Concentration', Cobalt: 90, RareEarths: 95, Lithium: 80, Copper: 50 },
        { subject: 'Geopolitical', Cobalt: 90, RareEarths: 85, Lithium: 50, Copper: 40 },
        { subject: 'Volatility', Cobalt: 70, RareEarths: 60, Lithium: 40, Copper: 30 },
        { subject: 'Demand Growth', Cobalt: 15, RareEarths: 8, Lithium: 30, Copper: 5 },
        { subject: 'Substitution', Cobalt: 50, RareEarths: 70, Lithium: 30, Copper: 20 }
    ]

    // Geographic concentration data
    const geographicData = [
        { material: 'Cobalt', country: 'DRC Congo', share: 75, flag: '🇨🇩', risk: 'Critical' },
        { material: 'Rare Earths', country: 'China', share: 70, flag: '🇨🇳', risk: 'Critical' },
        { material: 'Lithium', country: 'Australia/Chile', share: 75, flag: '🇦🇺', risk: 'Moderate' },
        { material: 'Graphite', country: 'China', share: 80, flag: '🇨🇳', risk: 'High' },
        { material: 'Nickel', country: 'Indonesia', share: 45, flag: '🇮🇩', risk: 'Moderate' },
        { material: 'Copper', country: 'Chile/Peru', share: 40, flag: '🇨🇱', risk: 'Low' },
        { material: 'Semiconductors', country: 'Taiwan', share: 92, flag: '🇹🇼', risk: 'Critical' }
    ]

    // Supply chain timeline/projections
    const supplyTimeline = [
        { year: 2024, lithiumSupply: 100, lithiumDemand: 85, copperSupply: 100, copperDemand: 92, cobaltSupply: 100, cobaltDemand: 78 },
        { year: 2025, lithiumSupply: 115, lithiumDemand: 105, copperSupply: 103, copperDemand: 98, cobaltSupply: 105, cobaltDemand: 88 },
        { year: 2027, lithiumSupply: 140, lithiumDemand: 160, copperSupply: 108, copperDemand: 115, cobaltSupply: 115, cobaltDemand: 105 },
        { year: 2030, lithiumSupply: 180, lithiumDemand: 280, copperSupply: 115, copperDemand: 145, cobaltSupply: 125, cobaltDemand: 130 },
        { year: 2035, lithiumSupply: 250, lithiumDemand: 500, copperSupply: 125, copperDemand: 200, cobaltSupply: 140, cobaltDemand: 155 }
    ]

    // Recycling potential data
    const recyclingData = [
        { material: 'Lithium', current: 5, potential: 95, gap: 90 },
        { material: 'Cobalt', current: 30, potential: 95, gap: 65 },
        { material: 'Nickel', current: 50, potential: 95, gap: 45 },
        { material: 'Copper', current: 35, potential: 98, gap: 63 },
        { material: 'Rare Earths', current: 1, potential: 80, gap: 79 },
        { material: 'Graphite', current: 0, potential: 90, gap: 90 }
    ]

    // Mitigation strategies
    const mitigationStrategies = [
        {
            material: 'Cobalt',
            strategies: ['LFP batteries (zero cobalt)', 'High-nickel cathodes', 'DRC diversification', 'Recycling mandates'],
            progress: 75,
            timeline: '2025-2027'
        },
        {
            material: 'Rare Earths',
            strategies: ['Alternative motor designs', 'Australia/US mining expansion', 'Recycling from e-waste'],
            progress: 35,
            timeline: '2027-2030'
        },
        {
            material: 'Semiconductors',
            strategies: ['US CHIPS Act ($52B)', 'EU Chips Act', 'Intel/TSMC Arizona fabs'],
            progress: 25,
            timeline: '2025-2028'
        },
        {
            material: 'Lithium',
            strategies: ['Direct lithium extraction', 'Argentina/Chile expansion', 'Seawater extraction R&D'],
            progress: 50,
            timeline: '2025-2028'
        }
    ]

    const getRiskColor = (risk) => {
        if (risk >= 65) return COLORS.critical
        if (risk >= 50) return COLORS.high
        if (risk >= 35) return COLORS.moderate
        return COLORS.low
    }

    const tabs = [
        { id: 'overview', label: '📊 Overview' },
        { id: 'risks', label: '⚠️ Risk Analysis' },
        { id: 'geography', label: '🌍 Geography' },
        { id: 'forecasts', label: '📈 Supply Forecasts' },
        { id: 'recycling', label: '♻️ Recycling' },
        { id: 'mitigation', label: '🛡️ Mitigation' }
    ]

    return (
        <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
            {/* Header */}
            <div style={{
                background: 'linear-gradient(135deg, #7c2d12 0%, #c2410c 50%, #ea580c 100%)',
                borderRadius: '20px',
                padding: '2.5rem',
                marginBottom: '2rem',
                color: 'white',
                position: 'relative',
                overflow: 'hidden'
            }}>
                <div style={{ position: 'absolute', top: 0, right: 0, opacity: 0.1, fontSize: '15rem', lineHeight: 1 }}>🏭</div>
                <h1 style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>
                    Supply Chain Intelligence
                </h1>
                <p style={{ fontSize: '1.1rem', opacity: 0.9, maxWidth: '600px' }}>
                    Critical materials risk analysis, geographic concentration, recycling potential,
                    and mitigation strategies for the EV transition.
                </p>
                <div style={{ display: 'flex', gap: '2rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', fontWeight: '700' }}>{evSupplyRisk.composite_score}/100</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>EV Supply Chain Risk</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', fontWeight: '700' }}>🔴 {evSupplyRisk.highest_risk_material}</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Highest Risk Material</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', fontWeight: '700' }}>7</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Critical Materials Tracked</div>
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
                                ? 'linear-gradient(135deg, #ea580c, #c2410c)'
                                : '#f1f5f9',
                            color: activeTab === tab.id ? 'white' : '#475569',
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
                    {/* Risk Score Cards */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
                        {riskChartData.map((item, idx) => (
                            <div key={idx} style={{
                                background: 'white',
                                borderRadius: '12px',
                                padding: '1.25rem',
                                boxShadow: '0 4px 15px rgba(0,0,0,0.08)',
                                borderTop: `4px solid ${getRiskColor(item.risk)}`
                            }}>
                                <div style={{ fontSize: '0.9rem', color: '#64748b', marginBottom: '0.25rem' }}>{item.name}</div>
                                <div style={{ fontSize: '2rem', fontWeight: '700', color: getRiskColor(item.risk) }}>{item.risk}</div>
                                <div style={{
                                    fontSize: '0.75rem',
                                    fontWeight: '600',
                                    color: getRiskColor(item.risk),
                                    background: `${getRiskColor(item.risk)}15`,
                                    padding: '0.25rem 0.5rem',
                                    borderRadius: '4px',
                                    display: 'inline-block',
                                    marginTop: '0.25rem'
                                }}>
                                    {item.level}
                                </div>
                                {item.reserveYears && (
                                    <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem' }}>
                                        {item.reserveYears} years reserves
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>

                    {/* Risk Breakdown Chart */}
                    <div style={{ background: 'white', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.08)', marginBottom: '2rem' }}>
                        <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '1rem' }}>Material Risk Composite Scores</h3>
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart data={riskChartData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis type="number" domain={[0, 100]} />
                                <YAxis type="category" dataKey="name" width={100} />
                                <Tooltip formatter={(value) => [`${value}/100`, 'Risk Score']} />
                                <Bar dataKey="risk" radius={[0, 8, 8, 0]}>
                                    {riskChartData.map((entry, index) => (
                                        <Cell key={index} fill={getRiskColor(entry.risk)} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Key Insights */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
                        <div style={{ background: '#fef2f2', borderRadius: '12px', padding: '1.5rem', borderLeft: '4px solid #ef4444' }}>
                            <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', color: '#991b1b' }}>🚨 Critical Risks</h4>
                            <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '0.9rem', color: '#7f1d1d' }}>
                                <li style={{ marginBottom: '0.5rem' }}>• 75% of cobalt from politically unstable DRC</li>
                                <li style={{ marginBottom: '0.5rem' }}>• 92% of advanced chips from Taiwan</li>
                                <li style={{ marginBottom: '0.5rem' }}>• China controls 70% of rare earth processing</li>
                                <li>• Graphite 80% concentrated in China</li>
                            </ul>
                        </div>
                        <div style={{ background: '#f0fdf4', borderRadius: '12px', padding: '1.5rem', borderLeft: '4px solid #22c55e' }}>
                            <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', color: '#166534' }}>✅ Positive Developments</h4>
                            <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '0.9rem', color: '#15803d' }}>
                                <li style={{ marginBottom: '0.5rem' }}>• LFP batteries eliminate cobalt (75% of China EVs)</li>
                                <li style={{ marginBottom: '0.5rem' }}>• Lithium price down 78% from peak (oversupply)</li>
                                <li style={{ marginBottom: '0.5rem' }}>• US CHIPS Act investing $52B in domestic fabs</li>
                                <li>• 95% of battery materials recyclable</li>
                            </ul>
                        </div>
                    </div>
                </div>
            )}

            {/* Risk Analysis Tab */}
            {activeTab === 'risks' && (
                <div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
                        {/* Radar Chart */}
                        <div style={{ background: 'white', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.08)' }}>
                            <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '1rem' }}>Risk Factor Comparison</h3>
                            <ResponsiveContainer width="100%" height={350}>
                                <RadarChart data={radarData}>
                                    <PolarGrid stroke="#e2e8f0" />
                                    <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11 }} />
                                    <PolarRadiusAxis angle={30} domain={[0, 100]} />
                                    <Radar name="Cobalt" dataKey="Cobalt" stroke={COLORS.critical} fill={COLORS.critical} fillOpacity={0.3} />
                                    <Radar name="Rare Earths" dataKey="RareEarths" stroke={COLORS.high} fill={COLORS.high} fillOpacity={0.3} />
                                    <Radar name="Lithium" dataKey="Lithium" stroke={COLORS.moderate} fill={COLORS.moderate} fillOpacity={0.3} />
                                    <Radar name="Copper" dataKey="Copper" stroke={COLORS.low} fill={COLORS.low} fillOpacity={0.3} />
                                    <Legend />
                                </RadarChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Risk Breakdown Table */}
                        <div style={{ background: 'white', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.08)' }}>
                            <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '1rem' }}>Detailed Risk Breakdown</h3>
                            <div style={{ overflowX: 'auto' }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                                            <th style={{ textAlign: 'left', padding: '0.75rem' }}>Material</th>
                                            <th style={{ textAlign: 'center', padding: '0.75rem' }}>Concentration</th>
                                            <th style={{ textAlign: 'center', padding: '0.75rem' }}>Geopolitical</th>
                                            <th style={{ textAlign: 'center', padding: '0.75rem' }}>Volatility</th>
                                            <th style={{ textAlign: 'center', padding: '0.75rem' }}>Growth</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {riskChartData.map((item, idx) => (
                                            <tr key={idx} style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                <td style={{ padding: '0.75rem', fontWeight: '500' }}>{item.name}</td>
                                                <td style={{ textAlign: 'center', padding: '0.75rem' }}>
                                                    <span style={{
                                                        background: item.concentration > 80 ? '#fee2e2' : item.concentration > 60 ? '#fef3c7' : '#dcfce7',
                                                        color: item.concentration > 80 ? '#991b1b' : item.concentration > 60 ? '#92400e' : '#166534',
                                                        padding: '0.25rem 0.5rem',
                                                        borderRadius: '4px',
                                                        fontWeight: '500'
                                                    }}>{item.concentration}%</span>
                                                </td>
                                                <td style={{ textAlign: 'center', padding: '0.75rem' }}>
                                                    <span style={{
                                                        background: item.geopolitical > 80 ? '#fee2e2' : item.geopolitical > 60 ? '#fef3c7' : '#dcfce7',
                                                        color: item.geopolitical > 80 ? '#991b1b' : item.geopolitical > 60 ? '#92400e' : '#166534',
                                                        padding: '0.25rem 0.5rem',
                                                        borderRadius: '4px',
                                                        fontWeight: '500'
                                                    }}>{item.geopolitical}%</span>
                                                </td>
                                                <td style={{ textAlign: 'center', padding: '0.75rem' }}>
                                                    <span style={{
                                                        background: item.volatility > 50 ? '#fee2e2' : item.volatility > 30 ? '#fef3c7' : '#dcfce7',
                                                        color: item.volatility > 50 ? '#991b1b' : item.volatility > 30 ? '#92400e' : '#166534',
                                                        padding: '0.25rem 0.5rem',
                                                        borderRadius: '4px',
                                                        fontWeight: '500'
                                                    }}>{item.volatility}%</span>
                                                </td>
                                                <td style={{ textAlign: 'center', padding: '0.75rem' }}>
                                                    <span style={{ fontWeight: '500' }}>{item.demandGrowth}%/yr</span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Geography Tab */}
            {activeTab === 'geography' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '1.5rem' }}>🌍 Geographic Concentration Risks</h2>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
                        {geographicData.map((item, idx) => (
                            <div key={idx} style={{
                                background: 'white',
                                borderRadius: '12px',
                                padding: '1.25rem',
                                boxShadow: '0 4px 15px rgba(0,0,0,0.08)',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '1rem'
                            }}>
                                <div style={{ fontSize: '3rem' }}>{item.flag}</div>
                                <div style={{ flex: 1 }}>
                                    <div style={{ fontWeight: '600', fontSize: '1rem' }}>{item.material}</div>
                                    <div style={{ color: '#64748b', fontSize: '0.9rem' }}>{item.country}</div>
                                    <div style={{ marginTop: '0.5rem' }}>
                                        <div style={{
                                            background: '#f1f5f9',
                                            height: '8px',
                                            borderRadius: '4px',
                                            overflow: 'hidden'
                                        }}>
                                            <div style={{
                                                width: `${item.share}%`,
                                                height: '100%',
                                                background: item.risk === 'Critical' ? COLORS.critical :
                                                    item.risk === 'High' ? COLORS.high :
                                                        item.risk === 'Moderate' ? COLORS.moderate : COLORS.low,
                                                borderRadius: '4px'
                                            }}></div>
                                        </div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.25rem' }}>
                                            <span style={{ fontSize: '0.75rem', fontWeight: '600' }}>{item.share}% market share</span>
                                            <span style={{
                                                fontSize: '0.7rem',
                                                fontWeight: '600',
                                                color: item.risk === 'Critical' ? COLORS.critical :
                                                    item.risk === 'High' ? COLORS.high :
                                                        item.risk === 'Moderate' ? COLORS.moderate : COLORS.low
                                            }}>{item.risk}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div style={{ background: '#fffbeb', borderRadius: '12px', padding: '1.5rem', borderLeft: '4px solid #f59e0b' }}>
                        <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', color: '#92400e' }}>⚠️ Key Geographic Risks</h4>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', fontSize: '0.9rem', color: '#78350f' }}>
                            <div>
                                <strong>Taiwan Semiconductor Risk:</strong> 92% of advanced chips. China tensions could disrupt entire automotive industry.
                            </div>
                            <div>
                                <strong>DRC Cobalt Risk:</strong> Artisanal mining, child labor concerns, political instability affect 75% of global supply.
                            </div>
                            <div>
                                <strong>China Processing Risk:</strong> Controls 80% of battery cell production and 70% of rare earth processing.
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Supply Forecasts Tab */}
            {activeTab === 'forecasts' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '1.5rem' }}>📈 Supply vs Demand Projections</h2>

                    <div style={{ background: 'white', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.08)', marginBottom: '2rem' }}>
                        <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '1rem' }}>Critical Material Supply-Demand Gap (Indexed to 2024=100)</h3>
                        <ResponsiveContainer width="100%" height={400}>
                            <ComposedChart data={supplyTimeline}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis dataKey="year" />
                                <YAxis domain={[0, 600]} />
                                <Tooltip />
                                <Legend />
                                <Area type="monotone" dataKey="lithiumDemand" name="Lithium Demand" fill="#ef444420" stroke="#ef4444" />
                                <Line type="monotone" dataKey="lithiumSupply" name="Lithium Supply" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" />
                                <Area type="monotone" dataKey="copperDemand" name="Copper Demand" fill="#f9731620" stroke="#f97316" />
                                <Line type="monotone" dataKey="copperSupply" name="Copper Supply" stroke="#f97316" strokeWidth={2} strokeDasharray="5 5" />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
                        <div style={{ background: '#fef2f2', borderRadius: '12px', padding: '1.25rem' }}>
                            <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', color: '#991b1b' }}>🔴 Lithium</h4>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#dc2626' }}>2027</div>
                            <div style={{ fontSize: '0.9rem', color: '#991b1b' }}>Projected deficit begins</div>
                            <div style={{ fontSize: '0.85rem', color: '#7f1d1d', marginTop: '0.5rem' }}>
                                Demand will exceed supply by 15% without new mining projects
                            </div>
                        </div>
                        <div style={{ background: '#fff7ed', borderRadius: '12px', padding: '1.25rem' }}>
                            <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', color: '#9a3412' }}>🟠 Copper</h4>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#ea580c' }}>2028</div>
                            <div style={{ fontSize: '0.9rem', color: '#9a3412' }}>Projected supply crunch</div>
                            <div style={{ fontSize: '0.85rem', color: '#7c2d12', marginTop: '0.5rem' }}>
                                EV + AI + renewables creating "copper collision"
                            </div>
                        </div>
                        <div style={{ background: '#f0fdf4', borderRadius: '12px', padding: '1.25rem' }}>
                            <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', color: '#166534' }}>🟢 Cobalt</h4>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#16a34a' }}>Declining</div>
                            <div style={{ fontSize: '0.9rem', color: '#166534' }}>Demand growth slowing</div>
                            <div style={{ fontSize: '0.85rem', color: '#15803d', marginTop: '0.5rem' }}>
                                LFP batteries reducing cobalt dependency by 30%+
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Recycling Tab */}
            {activeTab === 'recycling' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '1.5rem' }}>♻️ Recycling Potential & Gap Analysis</h2>

                    <div style={{ background: 'white', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.08)', marginBottom: '2rem' }}>
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart data={recyclingData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
                                <YAxis type="category" dataKey="material" width={80} />
                                <Tooltip formatter={(value) => `${value}%`} />
                                <Legend />
                                <Bar dataKey="current" name="Current Rate" stackId="a" fill="#ef4444" radius={[0, 0, 0, 0]} />
                                <Bar dataKey="gap" name="Untapped Potential" stackId="a" fill="#22c55e" radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                        {recyclingData.map((item, idx) => (
                            <div key={idx} style={{
                                background: 'white',
                                borderRadius: '12px',
                                padding: '1.25rem',
                                boxShadow: '0 4px 15px rgba(0,0,0,0.08)'
                            }}>
                                <div style={{ fontWeight: '600', marginBottom: '0.5rem' }}>{item.material}</div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                                    <span style={{ color: '#64748b' }}>Current: <span style={{ color: '#ef4444', fontWeight: '600' }}>{item.current}%</span></span>
                                    <span style={{ color: '#64748b' }}>Potential: <span style={{ color: '#22c55e', fontWeight: '600' }}>{item.potential}%</span></span>
                                </div>
                                <div style={{
                                    background: '#f1f5f9',
                                    height: '12px',
                                    borderRadius: '6px',
                                    overflow: 'hidden',
                                    position: 'relative'
                                }}>
                                    <div style={{
                                        width: `${item.potential}%`,
                                        height: '100%',
                                        background: '#22c55e40',
                                        position: 'absolute'
                                    }}></div>
                                    <div style={{
                                        width: `${item.current}%`,
                                        height: '100%',
                                        background: '#ef4444',
                                        position: 'absolute'
                                    }}></div>
                                </div>
                                <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.5rem' }}>
                                    {item.gap}% untapped recycling opportunity
                                </div>
                            </div>
                        ))}
                    </div>

                    <div style={{ background: '#f0fdf4', borderRadius: '12px', padding: '1.5rem', marginTop: '1.5rem' }}>
                        <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', color: '#166534' }}>💡 Why Recycling Matters</h4>
                        <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '0.9rem', color: '#15803d', display: 'grid', gap: '0.5rem' }}>
                            <li>• By 2030, ~1M tonnes of batteries will reach end-of-life annually</li>
                            <li>• Recycled materials cost 30-50% less than virgin mining</li>
                            <li>• EU mandating 70% battery recycling by 2030</li>
                            <li>• Urban mining could supply 10-20% of material needs by 2035</li>
                        </ul>
                    </div>
                </div>
            )}

            {/* Mitigation Tab */}
            {activeTab === 'mitigation' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '1.5rem' }}>🛡️ Risk Mitigation Strategies</h2>

                    <div style={{ display: 'grid', gap: '1.5rem' }}>
                        {mitigationStrategies.map((item, idx) => (
                            <div key={idx} style={{
                                background: 'white',
                                borderRadius: '16px',
                                padding: '1.5rem',
                                boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem', flexWrap: 'wrap', gap: '1rem' }}>
                                    <div>
                                        <h3 style={{ fontSize: '1.25rem', fontWeight: '700', marginBottom: '0.25rem' }}>{item.material}</h3>
                                        <span style={{ fontSize: '0.85rem', color: '#64748b' }}>Timeline: {item.timeline}</span>
                                    </div>
                                    <div style={{ textAlign: 'right' }}>
                                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: item.progress > 60 ? '#22c55e' : item.progress > 30 ? '#f59e0b' : '#ef4444' }}>
                                            {item.progress}%
                                        </div>
                                        <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Progress</div>
                                    </div>
                                </div>

                                <div style={{
                                    background: '#f1f5f9',
                                    height: '8px',
                                    borderRadius: '4px',
                                    overflow: 'hidden',
                                    marginBottom: '1rem'
                                }}>
                                    <div style={{
                                        width: `${item.progress}%`,
                                        height: '100%',
                                        background: item.progress > 60 ? '#22c55e' : item.progress > 30 ? '#f59e0b' : '#ef4444',
                                        borderRadius: '4px',
                                        transition: 'width 0.5s'
                                    }}></div>
                                </div>

                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                                    {item.strategies.map((strategy, sIdx) => (
                                        <span key={sIdx} style={{
                                            background: '#e0e7ff',
                                            color: '#4338ca',
                                            padding: '0.35rem 0.75rem',
                                            borderRadius: '20px',
                                            fontSize: '0.8rem',
                                            fontWeight: '500'
                                        }}>
                                            {strategy}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>

                    <div style={{ background: '#eff6ff', borderRadius: '12px', padding: '1.5rem', marginTop: '1.5rem' }}>
                        <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', color: '#1e40af' }}>📊 Key Policy Initiatives</h4>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', fontSize: '0.9rem' }}>
                            <div style={{ background: 'white', padding: '1rem', borderRadius: '8px' }}>
                                <div style={{ fontWeight: '600', color: '#1e3a8a' }}>US CHIPS Act</div>
                                <div style={{ color: '#3b82f6' }}>$52B semiconductor investment</div>
                            </div>
                            <div style={{ background: 'white', padding: '1rem', borderRadius: '8px' }}>
                                <div style={{ fontWeight: '600', color: '#1e3a8a' }}>EU Battery Regulation</div>
                                <div style={{ color: '#3b82f6' }}>70% recycling mandate by 2030</div>
                            </div>
                            <div style={{ background: 'white', padding: '1rem', borderRadius: '8px' }}>
                                <div style={{ fontWeight: '600', color: '#1e3a8a' }}>IRA Critical Minerals</div>
                                <div style={{ color: '#3b82f6' }}>Tax credits for domestic sourcing</div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
