import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import data from '../data/insights.json'

export default function Dashboard() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">📊 EV vs Gas Analysis Dashboard</h1>
                <p className="page-subtitle">Comprehensive ML-powered insights on the future of transportation</p>
            </header>

            {/* Key Stats */}
            <div className="stats-grid">
                {data.keyStats.map((stat, i) => (
                    <div className="stat-card" key={i}>
                        <div className="stat-icon">{stat.icon}</div>
                        <div className="stat-value">{stat.value}</div>
                        <div className="stat-label">{stat.label}</div>
                        <div className="stat-change">{stat.change}</div>
                    </div>
                ))}
            </div>

            {/* Charts */}
            <div className="grid-2">
                <div className="chart-container">
                    <h3 className="chart-title">🔋 Battery Cost Decline ($/kWh)</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={data.evAdoption.historical}>
                            <defs>
                                <linearGradient id="batteryGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis dataKey="year" stroke="#71717a" />
                            <YAxis stroke="#71717a" />
                            <Tooltip
                                contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                labelStyle={{ color: '#fafafa' }}
                            />
                            <Area type="monotone" dataKey="batteryCost" stroke="#3b82f6" fill="url(#batteryGradient)" strokeWidth={2} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                <div className="chart-container">
                    <h3 className="chart-title">📈 Global EV Sales (Millions)</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={data.evAdoption.historical}>
                            <defs>
                                <linearGradient id="salesGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis dataKey="year" stroke="#71717a" />
                            <YAxis stroke="#71717a" />
                            <Tooltip
                                contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                labelStyle={{ color: '#fafafa' }}
                            />
                            <Area type="monotone" dataKey="sales" stroke="#22c55e" fill="url(#salesGradient)" strokeWidth={2} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Quick Insights */}
            <div className="chart-container">
                <h3 className="chart-title">🎯 Key Insights at a Glance</h3>
                <div className="grid-3" style={{ marginTop: '1rem' }}>
                    <div className="card">
                        <h4 style={{ color: 'var(--accent-green)', marginBottom: '0.5rem' }}>🔥 Fire Safety</h4>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                            EVs have <strong style={{ color: 'var(--accent-green)' }}>98% fewer fires</strong> than gas cars (25 vs 1,530 per 100K vehicles)
                        </p>
                    </div>
                    <div className="card">
                        <h4 style={{ color: 'var(--accent-blue)', marginBottom: '0.5rem' }}>💰 Cost Savings</h4>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                            EVs cost <strong style={{ color: 'var(--accent-blue)' }}>$0.038/mile</strong> to charge at home vs $0.107/mile for gas
                        </p>
                    </div>
                    <div className="card">
                        <h4 style={{ color: 'var(--accent-purple)', marginBottom: '0.5rem' }}>🌍 Environment</h4>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                            EVs reduce lifetime CO2 by <strong style={{ color: 'var(--accent-purple)' }}>57-73%</strong> including manufacturing
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
