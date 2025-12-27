import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import data from '../data/insights.json'

export default function Costs() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">💰 Cost Analysis</h1>
                <p className="page-subtitle">Fuel costs, maintenance, and total cost of ownership</p>
            </header>

            <div className="grid-2">
                <div className="chart-container">
                    <h3 className="chart-title">⛽ Cost Per Mile</h3>
                    <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={data.costs.perMile}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis dataKey="type" stroke="#71717a" />
                            <YAxis stroke="#71717a" unit="$" />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                            <Bar dataKey="cost" radius={[8, 8, 0, 0]}>
                                {data.costs.perMile.map((entry, i) => (
                                    <Cell key={i} fill={entry.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                    <p style={{ color: 'var(--accent-green)', fontWeight: 600, marginTop: '1rem', textAlign: 'center' }}>
                        EV home charging is 65% cheaper per mile!
                    </p>
                </div>

                <div className="chart-container">
                    <h3 className="chart-title">🔧 Annual Savings</h3>
                    <div style={{ display: 'grid', gap: '1rem', marginTop: '1rem' }}>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-green)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span>Fuel Savings (12K mi/yr)</span>
                                <span style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: '1.25rem' }}>$800+</span>
                            </div>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-blue)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span>Maintenance Savings</span>
                                <span style={{ color: 'var(--accent-blue)', fontWeight: 700, fontSize: '1.25rem' }}>$660</span>
                            </div>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-purple)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span>Federal Tax Credit</span>
                                <span style={{ color: 'var(--accent-purple)', fontWeight: 700, fontSize: '1.25rem' }}>$7,500</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="chart-container">
                <h3 className="chart-title">📊 10-Year Total Cost of Ownership</h3>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>EV</th>
                            <th>Gas Car</th>
                            <th>Difference</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Purchase Price</td>
                            <td>$50,000</td>
                            <td>$45,000</td>
                            <td><span className="badge badge-red">+$5,000</span></td>
                        </tr>
                        <tr>
                            <td>Fuel (120K miles)</td>
                            <td>$4,500</td>
                            <td>$12,800</td>
                            <td><span className="badge badge-green">-$8,300</span></td>
                        </tr>
                        <tr>
                            <td>Maintenance</td>
                            <td>$5,000</td>
                            <td>$11,000</td>
                            <td><span className="badge badge-green">-$6,000</span></td>
                        </tr>
                        <tr>
                            <td>Tax Credit</td>
                            <td>-$7,500</td>
                            <td>$0</td>
                            <td><span className="badge badge-green">-$7,500</span></td>
                        </tr>
                        <tr style={{ fontWeight: 700, background: 'var(--bg-hover)' }}>
                            <td>TOTAL SAVINGS</td>
                            <td colSpan="2"></td>
                            <td><span style={{ color: 'var(--accent-green)', fontSize: '1.25rem' }}>-$20,800</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    )
}
