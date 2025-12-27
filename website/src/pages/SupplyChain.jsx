import data from '../data/insights.json'

export default function SupplyChain() {
    const getRiskColor = (risk) => {
        if (risk >= 8) return '#ef4444'
        if (risk >= 5) return '#f97316'
        return '#22c55e'
    }

    const getRiskLabel = (risk) => {
        if (risk >= 8) return 'CRITICAL'
        if (risk >= 5) return 'HIGH'
        return 'LOW'
    }

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🏭 Supply Chain Analysis</h1>
                <p className="page-subtitle">Battery materials, mining risks, and recycling potential</p>
            </header>

            <div className="chart-container">
                <h3 className="chart-title">⚠️ Critical Battery Materials</h3>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Material</th>
                            <th>kg per EV</th>
                            <th>Top Producer</th>
                            <th>Risk Level</th>
                            <th>Recycling Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.supplyChain.map((item, i) => (
                            <tr key={i}>
                                <td><strong>{item.material}</strong></td>
                                <td>{item.kgPerEV} kg</td>
                                <td>{item.topProducer}</td>
                                <td>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                        <div className="risk-bar" style={{ width: '80px' }}>
                                            <div className="risk-fill" style={{
                                                width: `${item.risk * 10}%`,
                                                background: getRiskColor(item.risk)
                                            }} />
                                        </div>
                                        <span style={{ color: getRiskColor(item.risk), fontWeight: 600, fontSize: '0.75rem' }}>
                                            {getRiskLabel(item.risk)}
                                        </span>
                                    </div>
                                </td>
                                <td>{item.recycling}%</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="grid-2">
                <div className="chart-container">
                    <h3 className="chart-title">🚨 Key Risks</h3>
                    <div style={{ display: 'grid', gap: '1rem', marginTop: '1rem' }}>
                        <div className="card" style={{ borderLeft: '3px solid #ef4444' }}>
                            <h4>Cobalt: DRC Congo</h4>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                                75% of global cobalt comes from one unstable country
                            </p>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid #ef4444' }}>
                            <h4>Rare Earths: China</h4>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                                70% dominance, almost no recycling (1%)
                            </p>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid #f97316' }}>
                            <h4>Batteries: China</h4>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                                80% of global battery production passes through China
                            </p>
                        </div>
                    </div>
                </div>

                <div className="chart-container">
                    <h3 className="chart-title">✅ Positive Trends</h3>
                    <div style={{ display: 'grid', gap: '1rem', marginTop: '1rem' }}>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-green)' }}>
                            <h4>LFP Batteries (No Cobalt)</h4>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                                75% of Chinese EVs now use cobalt-free LFP chemistry
                            </p>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-green)' }}>
                            <h4>Lithium Oversupply</h4>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                                Price crashed from $70K to $15K/tonne (oversupply)
                            </p>
                        </div>
                        <div className="card" style={{ borderLeft: '3px solid var(--accent-blue)' }}>
                            <h4>Recycling Potential</h4>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                                95% of battery materials are theoretically recyclable
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
