import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, ComposedChart, Bar } from 'recharts'
import data from '../data/insights.json'

export default function Predictions() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">📈 ML Predictions (2025-2050)</h1>
                <p className="page-subtitle">Machine learning forecasts using XGBoost, LightGBM, CatBoost, and PyTorch</p>
            </header>

            <div className="grid-2">
                <div className="chart-container">
                    <h3 className="chart-title">🚗 Fleet Composition Forecast</h3>
                    <ResponsiveContainer width="100%" height={350}>
                        <AreaChart data={data.evAdoption.predictions}>
                            <defs>
                                <linearGradient id="evGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.8} />
                                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0.1} />
                                </linearGradient>
                                <linearGradient id="hybridGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis dataKey="year" stroke="#71717a" />
                            <YAxis stroke="#71717a" unit="%" />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                            <Area type="monotone" dataKey="evPct" name="EV %" stackId="1" stroke="#22c55e" fill="url(#evGrad)" />
                            <Area type="monotone" dataKey="hybridPct" name="Hybrid %" stackId="1" stroke="#3b82f6" fill="url(#hybridGrad)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                <div className="chart-container">
                    <h3 className="chart-title">⚡ Infrastructure Crossover</h3>
                    <ResponsiveContainer width="100%" height={350}>
                        <LineChart data={data.evAdoption.predictions}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis dataKey="year" stroke="#71717a" />
                            <YAxis stroke="#71717a" />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                            <Line type="monotone" dataKey="chargingStations" name="EV Stations" stroke="#22c55e" strokeWidth={3} dot={false} />
                            <Line type="monotone" dataKey="gasStations" name="Gas Stations" stroke="#ef4444" strokeWidth={3} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.5rem' }}>
                        ⚡ Crossover point: ~2035 when EV stations exceed gas stations
                    </p>
                </div>
            </div>

            {/* Predictions Table */}
            <div className="chart-container">
                <h3 className="chart-title">📊 Prediction Milestones</h3>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Year</th>
                            <th>EV %</th>
                            <th>EVs on Road (M)</th>
                            <th>Battery Cost</th>
                            <th>EV Stations</th>
                            <th>Gas Stations</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.evAdoption.predictions.map((row, i) => (
                            <tr key={i}>
                                <td><strong>{row.year}</strong></td>
                                <td><span className="badge badge-green">{row.evPct}%</span></td>
                                <td>{row.evStock}M</td>
                                <td>${row.batteryCost}/kWh</td>
                                <td>{(row.chargingStations / 1000).toFixed(0)}K</td>
                                <td>{(row.gasStations / 1000).toFixed(0)}K</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
