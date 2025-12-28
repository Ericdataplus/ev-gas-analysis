import { useState } from 'react'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend, AreaChart, Area } from 'recharts'
import analysisData from '../data/global_infrastructure.json'

const COLORS = ['#6366f1', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#14b8a6']

export default function GlobalInfrastructure() {
    const [activeTab, setActiveTab] = useState('overview')
    const sectors = analysisData?.sectors || {}

    const tabs = [
        { id: 'overview', label: '📊 Overview', color: '#6366f1' },
        { id: 'energy', label: '⚡ Energy', color: '#f97316' },
        { id: 'infrastructure', label: '🏗️ Infrastructure', color: '#22c55e' },
        { id: 'automotive', label: '🚗 Automotive', color: '#ef4444' },
        { id: 'aerospace', label: '✈️ Aerospace', color: '#8b5cf6' },
        { id: 'tech', label: '💻 Tech', color: '#06b6d4' },
        { id: 'industry', label: '🏭 Industry', color: '#ec4899' },
    ]

    // Data extraction
    const energy = sectors.energy || {}
    const infra = sectors.infrastructure || {}
    const auto = sectors.automotive || {}
    const aero = sectors.aerospace || {}
    const tech = sectors.tech || {}
    const industry = sectors.heavy_industry || {}

    // Energy mix chart
    const energyMixData = Object.entries(energy.energy_mix_2024 || {}).map(([name, data]) => ({
        name: name.replace('_', ' '),
        share: data.share_pct,
        emissions: data.emissions_gt
    }))

    // Fuel consumption
    const fuelData = Object.entries(energy.fuel_consumption || {}).map(([name, data]) => ({
        name: name.replace(/_/g, ' '),
        share: data.share_pct
    }))

    // Automakers
    const automakerData = Object.entries(auto.automakers || {}).map(([name, data]) => ({
        name,
        production: data.production_m,
        ev: Math.round(data.ev_share * 100)
    })).sort((a, b) => b.production - a.production)

    // Semiconductors
    const semiData = Object.entries(tech.semiconductors || {}).map(([name, data]) => ({
        name,
        share: Math.round(data.market_share * 100),
        revenue: data.revenue_b
    }))

    // Steel by country
    const steelData = Object.entries(industry.steel || {}).map(([name, data]) => ({
        name: name.replace('_', ' '),
        production: data.production_mt
    }))

    // Grid capacity
    const gridData = Object.entries(infra.grid_capacity || {}).map(([name, data]) => ({
        name,
        capacity: data.capacity_gw,
        renewable: data.renewable_pct
    }))

    return (
        <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
            {/* Hero */}
            <div style={{
                background: 'linear-gradient(135deg, #0a0a1a 0%, #1a2e1a 30%, #1a3a3a 60%, #2d1b4d 100%)',
                borderRadius: '24px', padding: '3rem', marginBottom: '2rem', color: 'white',
                position: 'relative', overflow: 'hidden'
            }}>
                <div style={{ position: 'absolute', top: -50, right: -50, fontSize: '18rem', opacity: 0.05 }}>🌍</div>
                <div style={{
                    fontSize: '0.8rem', background: 'linear-gradient(135deg, #22c55e, #16a34a)',
                    display: 'inline-block', padding: '0.25rem 1rem', borderRadius: '20px',
                    marginBottom: '1rem', fontWeight: '700'
                }}>
                    🌐 COMPREHENSIVE GLOBAL ANALYSIS
                </div>
                <h1 style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>
                    🌍 Global Energy & Infrastructure
                </h1>
                <p style={{ fontSize: '1.1rem', opacity: 0.9, maxWidth: '800px' }}>
                    Complete analysis of energy production, infrastructure networks, automotive, aerospace,
                    tech manufacturing, and heavy industry across all major economies.
                </p>
                <div style={{ display: 'flex', gap: '1rem', marginTop: '2rem', flexWrap: 'wrap' }}>
                    {[
                        { value: '7', label: 'Sectors' },
                        { value: '600+ EJ', label: 'Global Energy' },
                        { value: '96M', label: 'Vehicles/Year' },
                        { value: '2900 GW', label: 'Grid Capacity' },
                        { value: '1.9B MT', label: 'Steel Output' },
                    ].map((stat, i) => (
                        <div key={i} style={{ background: 'rgba(255,255,255,0.1)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700' }}>{stat.value}</div>
                            <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>{stat.label}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Tabs */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
                {tabs.map(tab => (
                    <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
                        padding: '0.75rem 1.25rem', borderRadius: '10px', border: 'none',
                        background: activeTab === tab.id ? `linear-gradient(135deg, ${tab.color}, ${tab.color}dd)` : 'var(--bg-card)',
                        color: activeTab === tab.id ? 'white' : 'var(--text-secondary)',
                        fontWeight: '600', cursor: 'pointer'
                    }}>{tab.label}</button>
                ))}
            </div>

            {/* OVERVIEW */}
            {activeTab === 'overview' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>📊 Global Industry Overview</h2>
                    <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
                        <div className="chart-container">
                            <h3 className="chart-title">Global Energy Mix 2024 (%)</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <PieChart>
                                    <Pie data={energyMixData} dataKey="share" nameKey="name" cx="50%" cy="50%" outerRadius={100} label={e => `${e.name}: ${e.share}%`}>
                                        {energyMixData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                                    </Pie>
                                    <Tooltip />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="chart-container">
                            <h3 className="chart-title">Fuel Consumption by Sector (%)</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={fuelData} layout="vertical">
                                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                    <XAxis type="number" stroke="#71717a" />
                                    <YAxis dataKey="name" type="category" width={120} stroke="#71717a" fontSize={10} />
                                    <Tooltip />
                                    <Bar dataKey="share" fill="#f97316" radius={[0, 8, 8, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                    <div className="card" style={{ padding: '1.5rem' }}>
                        <h3 style={{ marginBottom: '1rem' }}>🎯 Key Insights</h3>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '0.75rem' }}>
                            <div style={{ padding: '1rem', background: 'rgba(249, 115, 22, 0.1)', borderRadius: '8px', borderLeft: '3px solid #f97316' }}>
                                <strong>Ground transport uses 70%</strong> of petroleum fuel (cars + trucks)
                            </div>
                            <div style={{ padding: '1rem', background: 'rgba(34, 197, 94, 0.1)', borderRadius: '8px', borderLeft: '3px solid #22c55e' }}>
                                <strong>Renewables at 15%</strong> and growing rapidly (wind + solar doubling every 3-4 years)
                            </div>
                            <div style={{ padding: '1rem', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '8px', borderLeft: '3px solid #6366f1' }}>
                                <strong>China dominates</strong> manufacturing: 32% of cars, 54% of steel, 60% of aluminum
                            </div>
                            <div style={{ padding: '1rem', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '8px', borderLeft: '3px solid #8b5cf6' }}>
                                <strong>Aviation only 7%</strong> of fuel use - ground transport is the priority for electrification
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* ENERGY */}
            {activeTab === 'energy' && energyMixData.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>⚡ Global Energy Production</h2>
                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">Energy Mix & Emissions</h3>
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart data={energyMixData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="name" stroke="#71717a" />
                                <YAxis yAxisId="left" stroke="#71717a" />
                                <YAxis yAxisId="right" orientation="right" stroke="#71717a" />
                                <Tooltip />
                                <Legend />
                                <Bar yAxisId="left" dataKey="share" name="Share %" fill="#6366f1" radius={[4, 4, 0, 0]} />
                                <Bar yAxisId="right" dataKey="emissions" name="Emissions GT" fill="#ef4444" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="card" style={{ padding: '1.5rem' }}>
                        <h4 style={{ marginBottom: '1rem' }}>Top Oil Producers</h4>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.75rem' }}>
                            {Object.entries(energy.oil_producers || {}).map(([name, data], i) => (
                                <div key={name} style={{ padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px', borderTop: `3px solid ${COLORS[i]}` }}>
                                    <div style={{ fontWeight: '600' }}>{name.replace('_', ' ')}</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: '700', color: COLORS[i] }}>{data.production_mbd} mbd</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* INFRASTRUCTURE */}
            {activeTab === 'infrastructure' && gridData.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🏗️ Global Infrastructure</h2>
                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">Power Grid Capacity by Region (GW)</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={gridData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="name" stroke="#71717a" />
                                <YAxis stroke="#71717a" />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="capacity" name="Capacity GW" fill="#22c55e" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="renewable" name="Renewable %" fill="#6366f1" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="card" style={{ padding: '1.5rem' }}>
                        <h4 style={{ marginBottom: '1rem' }}>EV Charging Networks</h4>
                        {Object.entries(infra.charging_infra || {}).map(([name, data], i) => (
                            <div key={name} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem 0', borderBottom: '1px solid var(--border-color)' }}>
                                <span>{name.replace('_', ' ')}</span>
                                <span style={{ fontWeight: '600', color: COLORS[i] }}>{data.stations.toLocaleString()} stations</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* AUTOMOTIVE */}
            {activeTab === 'automotive' && automakerData.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🚗 Automotive Manufacturing</h2>
                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">Top Automakers (Millions of Vehicles)</h3>
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart data={automakerData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" />
                                <YAxis dataKey="name" type="category" width={100} stroke="#71717a" />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="production" name="Production (M)" fill="#ef4444" radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* AEROSPACE */}
            {activeTab === 'aerospace' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>✈️ Aerospace Manufacturing</h2>
                    <div className="grid-2">
                        <div className="card" style={{ padding: '1.5rem' }}>
                            <h4 style={{ marginBottom: '1rem' }}>Commercial Aircraft</h4>
                            {Object.entries(aero.commercial_aircraft || {}).map(([name, data], i) => (
                                <div key={name} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-color)' }}>
                                    <span>{name}</span>
                                    <span style={{ color: COLORS[i] }}>{data.backlog.toLocaleString()} backlog</span>
                                </div>
                            ))}
                        </div>
                        <div className="card" style={{ padding: '1.5rem' }}>
                            <h4 style={{ marginBottom: '1rem' }}>Space Industry</h4>
                            {Object.entries(aero.space || {}).map(([name, data], i) => (
                                <div key={name} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-color)' }}>
                                    <span>{name.replace('_', ' ')}</span>
                                    <span style={{ color: COLORS[i] }}>{data.launches_2024} launches</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* TECH */}
            {activeTab === 'tech' && semiData.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>💻 Tech Manufacturing</h2>
                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">Semiconductor Market Share (%)</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={semiData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="name" stroke="#71717a" />
                                <YAxis stroke="#71717a" />
                                <Tooltip />
                                <Bar dataKey="share" fill="#06b6d4" radius={[8, 8, 0, 0]}>
                                    {semiData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* INDUSTRY */}
            {activeTab === 'industry' && steelData.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🏭 Heavy Industry</h2>
                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">Steel Production (Million Tonnes)</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={steelData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="name" stroke="#71717a" />
                                <YAxis stroke="#71717a" />
                                <Tooltip />
                                <Bar dataKey="production" fill="#ec4899" radius={[8, 8, 0, 0]}>
                                    {steelData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="card" style={{ padding: '1.5rem' }}>
                        <h4 style={{ marginBottom: '1rem' }}>Critical Minerals</h4>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.75rem' }}>
                            {Object.entries(industry.minerals || {}).map(([name, data], i) => (
                                <div key={name} style={{ padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{name.replace('_', ' ')}</div>
                                    <div style={{ fontWeight: '700', color: COLORS[i] }}>{data.production_kt} kt</div>
                                    <div style={{ fontSize: '0.75rem' }}>Top: {data.top_producer}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
