import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import ChartModal from '../components/ChartModal'
import data from '../data/insights.json'

export default function Costs() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">💰 Cost Analysis</h1>
                <p className="page-subtitle">Fuel costs, maintenance, and total cost of ownership</p>
            </header>

            <div className="grid-2">
                <ChartModal
                    title="⛽ Cost Per Mile Comparison"
                    insight="EV home charging costs just $0.038 per mile - 65% cheaper than gas at $0.107/mile. DC fast charging costs about the same as gas. Over 12,000 annual miles, that's $800+ in yearly fuel savings, plus $660 in maintenance savings!"
                >
                    <ResponsiveContainer width="100%" height="100%">
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
                </ChartModal>

                <div className="chart-container">
                    <h3 className="chart-title">🔧 Annual Savings</h3>
                    <div style={{ display: 'grid', gap: '0.75rem', marginTop: '0.5rem' }}>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-green)', padding: '0.75rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ fontSize: '0.9rem' }}>Fuel Savings (12K mi/yr)</span>
                                <span style={{ color: 'var(--accent-green)', fontWeight: 700 }}>$800+</span>
                            </div>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-blue)', padding: '0.75rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ fontSize: '0.9rem' }}>Maintenance Savings</span>
                                <span style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>$660</span>
                            </div>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-purple)', padding: '0.75rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ fontSize: '0.9rem' }}>Federal Tax Credit</span>
                                <span style={{ color: 'var(--accent-purple)', fontWeight: 700 }}>$7,500</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="chart-container">
                <h3 className="chart-title">📊 10-Year Total Cost of Ownership</h3>
                <table className="data-table" style={{ fontSize: '0.9rem' }}>
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
                            <td><span style={{ color: 'var(--accent-green)', fontSize: '1.1rem' }}>$20,800</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    )
}
