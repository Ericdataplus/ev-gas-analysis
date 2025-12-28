import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
    AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis,
    CartesianGrid, Tooltip, ResponsiveContainer, Cell, ComposedChart, Legend
} from 'recharts'

// Import all analysis data
import insights from '../data/insights.json'

const COLORS = ['#6366f1', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899']

// Try to load additional data
let enterpriseData = null
let cuttingEdgeData = null
let professionalData = null

try { enterpriseData = require('../data/enterprise_analysis.json') } catch { }
try { cuttingEdgeData = require('../data/cutting_edge_ml.json') } catch { }
try { professionalData = require('../data/professional_analysis.json') } catch { }

export default function Dashboard() {
    const [hoveredCard, setHoveredCard] = useState(null)

    // Hero stats from various sources
    const heroStats = [
        { icon: '🚗', value: '17.8M', label: 'Global EV Sales 2024', trend: '+38%', color: '#22c55e' },
        { icon: '🔋', value: '$100', label: 'Battery $/kWh', trend: '-15% YoY', color: '#6366f1' },
        { icon: '⚡', value: '192K', label: 'US Charging Stations', trend: '+45%', color: '#f97316' },
        { icon: '🏢', value: '25', label: 'Major OEMs Analyzed', trend: '100+ models', color: '#8b5cf6' },
    ]

    // ML analysis summary
    const mlSummary = [
        { name: 'Breakthrough Insights', models: 7, icon: '🔮', path: '/breakthrough', color: '#ec4899' },
        { name: '56 Granular Questions', models: 56, icon: '🔬', path: '/granular', color: '#06b6d4' },
        { name: 'Professional Suite', models: 18, icon: '🏆', path: '/professional', color: '#22c55e' },
        { name: 'Enterprise Analysis', models: 100, icon: '🏢', path: '/enterprise', color: '#f97316' },
        { name: 'Cutting-Edge ML', models: 10, icon: '🧠', path: '/cutting-edge', color: '#8b5cf6' },
    ]

    // Market projections
    const marketProjections = [
        { year: 2024, sales: 17.8, share: 18 },
        { year: 2025, sales: 22, share: 22 },
        { year: 2026, sales: 27, share: 27 },
        { year: 2027, sales: 33, share: 32 },
        { year: 2028, sales: 40, share: 38 },
        { year: 2029, sales: 48, share: 44 },
        { year: 2030, sales: 60, share: 50 },
    ]

    // Battery technology roadmap
    const batteryRoadmap = [
        { year: 2024, cost: 100, density: 270, range: 300 },
        { year: 2026, cost: 75, density: 320, range: 380 },
        { year: 2028, cost: 55, density: 380, range: 450 },
        { year: 2030, cost: 45, density: 450, range: 550 },
    ]

    // Regional market data
    const regionalData = [
        { region: 'China', sales: 10.2, share: 35, color: '#ef4444' },
        { region: 'Europe', sales: 3.8, share: 24, color: '#6366f1' },
        { region: 'USA', sales: 1.8, share: 9, color: '#22c55e' },
        { region: 'Others', sales: 2.0, share: 8, color: '#f97316' },
    ]

    // Key findings
    const keyFindings = [
        { stat: '98%', label: 'Fewer fires than gas cars', icon: '🔥', color: '#22c55e' },
        { stat: '$0.038', label: 'Cost per mile (vs $0.107 gas)', icon: '💰', color: '#6366f1' },
        { stat: '57-73%', label: 'Lower lifetime CO2 emissions', icon: '🌍', color: '#8b5cf6' },
        { stat: '40.5%', label: 'Predicted 2030 market share', icon: '📈', color: '#f97316' },
    ]

    return (
        <div style={{ padding: '1.5rem' }}>
            {/* Hero Section */}
            <div style={{
                background: 'linear-gradient(135deg, #0c0a20 0%, #1a1744 30%, #2d1b69 60%, #4c1d95 100%)',
                borderRadius: '24px',
                padding: '3rem',
                marginBottom: '2rem',
                position: 'relative',
                overflow: 'hidden'
            }}>
                {/* Background pattern */}
                <div style={{
                    position: 'absolute',
                    top: 0, left: 0, right: 0, bottom: 0,
                    background: 'radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 40%), radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 40%)',
                    pointerEvents: 'none'
                }} />

                <div style={{ position: 'relative', zIndex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                        <span style={{ fontSize: '2.5rem' }}>⚡</span>
                        <div>
                            <h1 style={{ fontSize: '2.5rem', fontWeight: '800', color: 'white', margin: 0, lineHeight: 1.1 }}>
                                Energy Transition Intelligence
                            </h1>
                            <p style={{ color: 'rgba(255,255,255,0.7)', margin: '0.5rem 0 0', fontSize: '1.1rem' }}>
                                100+ ML models • 200+ insights • Real-time analysis
                            </p>
                        </div>
                    </div>

                    {/* Hero Stats Grid */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginTop: '2rem' }}>
                        {heroStats.map((stat, i) => (
                            <div
                                key={i}
                                style={{
                                    background: 'rgba(255,255,255,0.08)',
                                    backdropFilter: 'blur(10px)',
                                    borderRadius: '16px',
                                    padding: '1.25rem',
                                    border: '1px solid rgba(255,255,255,0.1)',
                                    transition: 'all 0.3s ease',
                                    cursor: 'pointer',
                                    transform: hoveredCard === `hero-${i}` ? 'translateY(-4px)' : 'none',
                                }}
                                onMouseEnter={() => setHoveredCard(`hero-${i}`)}
                                onMouseLeave={() => setHoveredCard(null)}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                    <span style={{ fontSize: '2rem' }}>{stat.icon}</span>
                                    <span style={{
                                        background: `${stat.color}30`,
                                        color: stat.color,
                                        padding: '0.25rem 0.5rem',
                                        borderRadius: '6px',
                                        fontSize: '0.75rem',
                                        fontWeight: '600'
                                    }}>
                                        {stat.trend}
                                    </span>
                                </div>
                                <div style={{ color: 'white', fontSize: '2rem', fontWeight: '700', margin: '0.5rem 0' }}>
                                    {stat.value}
                                </div>
                                <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.85rem' }}>
                                    {stat.label}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* ML Analysis Suites */}
            <div style={{ marginBottom: '2rem' }}>
                <h2 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span>🤖</span> ML Analysis Suites
                </h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem' }}>
                    {mlSummary.map((suite, i) => (
                        <NavLink
                            key={i}
                            to={suite.path}
                            style={{
                                textDecoration: 'none',
                                background: 'var(--bg-card)',
                                borderRadius: '12px',
                                padding: '1.25rem',
                                borderLeft: `4px solid ${suite.color}`,
                                transition: 'all 0.2s',
                                display: 'block'
                            }}
                            className="card-hover"
                        >
                            <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>{suite.icon}</div>
                            <div style={{ fontSize: '0.9rem', fontWeight: '600', color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                                {suite.name}
                            </div>
                            <div style={{ fontSize: '1.25rem', fontWeight: '700', color: suite.color }}>
                                {suite.models} {suite.models > 20 ? 'analyses' : 'models'}
                            </div>
                        </NavLink>
                    ))}
                </div>
            </div>

            {/* Charts Row */}
            <div className="grid-2" style={{ marginBottom: '2rem' }}>
                {/* Market Projections */}
                <div className="chart-container">
                    <h3 className="chart-title">📈 Global EV Market Projection</h3>
                    <ResponsiveContainer width="100%" height={280}>
                        <ComposedChart data={marketProjections}>
                            <defs>
                                <linearGradient id="salesGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis dataKey="year" stroke="#71717a" />
                            <YAxis yAxisId="left" stroke="#71717a" />
                            <YAxis yAxisId="right" orientation="right" stroke="#71717a" tickFormatter={v => `${v}%`} />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                            <Legend />
                            <Area yAxisId="left" type="monotone" dataKey="sales" name="Sales (M)" fill="url(#salesGrad)" stroke="#22c55e" strokeWidth={2} />
                            <Line yAxisId="right" type="monotone" dataKey="share" name="Market %" stroke="#6366f1" strokeWidth={2} dot={{ fill: '#6366f1' }} />
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>

                {/* Battery Roadmap */}
                <div className="chart-container">
                    <h3 className="chart-title">🔋 Battery Technology Roadmap</h3>
                    <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={batteryRoadmap}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis dataKey="year" stroke="#71717a" />
                            <YAxis stroke="#71717a" />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                            <Legend />
                            <Bar dataKey="cost" name="Cost $/kWh" fill="#ef4444" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="range" name="Range (mi)" fill="#22c55e" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Key Findings Row */}
            <div style={{ marginBottom: '2rem' }}>
                <h2 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span>🎯</span> Key Findings
                </h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                    {keyFindings.map((finding, i) => (
                        <div
                            key={i}
                            className="card"
                            style={{
                                padding: '1.5rem',
                                textAlign: 'center',
                                borderTop: `4px solid ${finding.color}`
                            }}
                        >
                            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>{finding.icon}</div>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: finding.color }}>
                                {finding.stat}
                            </div>
                            <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                                {finding.label}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Regional + Categories Row */}
            <div className="grid-2">
                {/* Regional Breakdown */}
                <div className="chart-container">
                    <h3 className="chart-title">🌍 Regional Market Share 2024</h3>
                    <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={regionalData} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis type="number" stroke="#71717a" tickFormatter={v => `${v}M`} />
                            <YAxis dataKey="region" type="category" stroke="#71717a" width={60} />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} formatter={v => `${v}M vehicles`} />
                            <Bar dataKey="sales" radius={[0, 8, 8, 0]}>
                                {regionalData.map((entry, i) => (
                                    <Cell key={i} fill={entry.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Quick Navigation */}
                <div className="card" style={{ padding: '1.5rem' }}>
                    <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span>🧭</span> Explore More
                    </h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                        {[
                            { path: '/costs', label: 'TCO Analysis', icon: '💰' },
                            { path: '/safety', label: 'Safety Stats', icon: '🛡️' },
                            { path: '/supply-chain', label: 'Supply Chain', icon: '🔗' },
                            { path: '/batteries', label: 'Battery Tech', icon: '🔋' },
                            { path: '/market', label: '2025 Market', icon: '🏆' },
                            { path: '/predictions', label: 'Predictions', icon: '🔮' },
                        ].map((item, i) => (
                            <NavLink
                                key={i}
                                to={item.path}
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '0.5rem',
                                    padding: '0.75rem 1rem',
                                    background: 'var(--bg-tertiary)',
                                    borderRadius: '8px',
                                    textDecoration: 'none',
                                    color: 'var(--text-primary)',
                                    transition: 'all 0.2s'
                                }}
                                className="nav-card"
                            >
                                <span style={{ fontSize: '1.25rem' }}>{item.icon}</span>
                                <span style={{ fontWeight: '500' }}>{item.label}</span>
                            </NavLink>
                        ))}
                    </div>
                </div>
            </div>

            {/* Footer Stats */}
            <div style={{
                marginTop: '2rem',
                padding: '1.5rem',
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1))',
                borderRadius: '16px',
                display: 'flex',
                justifyContent: 'space-around',
                flexWrap: 'wrap',
                gap: '1rem'
            }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#6366f1' }}>100+</div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>ML Models</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#22c55e' }}>200+</div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Insights</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#f97316' }}>25</div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Manufacturers</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#8b5cf6' }}>7</div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Regions</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#ec4899' }}>2035</div>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Projections</div>
                </div>
            </div>
        </div>
    )
}
