import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import ChartModal from '../components/ChartModal'
import data from '../data/insights.json'

export default function Solar() {
    const adoptionData = [
        { year: 2016, homes: 1.3 },
        { year: 2018, homes: 2.0 },
        { year: 2020, homes: 2.7 },
        { year: 2022, homes: 3.6 },
        { year: 2024, homes: 4.7 },
        { year: 2030, homes: 10.0 },
    ]

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">☀️ Residential Solar</h1>
                <p className="page-subtitle">Home solar adoption, economics, and the path to energy independence</p>
            </header>

            {/* Key Stats */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon">🏠</div>
                    <div className="stat-value">{data.residentialSolar.adoption2024.percentOfHomes}%</div>
                    <div className="stat-label">US Homes with Solar</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">📈</div>
                    <div className="stat-value">{(data.residentialSolar.adoption2024.totalHouseholds / 1000000).toFixed(1)}M</div>
                    <div className="stat-label">Solar Households</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">💰</div>
                    <div className="stat-value">{data.residentialSolar.economics.federalTaxCredit}%</div>
                    <div className="stat-label">Federal Tax Credit</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">⏱️</div>
                    <div className="stat-value">{data.residentialSolar.economics.paybackPeriodYears.average} yrs</div>
                    <div className="stat-label">Avg Payback Period</div>
                </div>
            </div>

            <div className="grid-2">
                <ChartModal
                    title="📈 Solar Adoption Growth (Millions of Homes)"
                    insight={`Solar adoption grew ${data.residentialSolar.adoption2024.growth2016To2024}% from 2016-2024. Projections show 15% of US homes will have solar by 2030 - more than doubling current adoption.`}
                >
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={adoptionData}>
                            <defs>
                                <linearGradient id="solarGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#eab308" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#eab308" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                            <XAxis dataKey="year" stroke="#71717a" />
                            <YAxis stroke="#71717a" unit="M" />
                            <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                            <Area type="monotone" dataKey="homes" stroke="#eab308" fill="url(#solarGrad)" strokeWidth={2} />
                        </AreaChart>
                    </ResponsiveContainer>
                </ChartModal>

                <div className="chart-container">
                    <h3 className="chart-title">💰 Solar Economics</h3>
                    <div style={{ display: 'grid', gap: '0.5rem', marginTop: '0.5rem' }}>
                        <div className="card" style={{ padding: '0.75rem', display: 'flex', justifyContent: 'space-between' }}>
                            <span>Average System Cost</span>
                            <span style={{ fontWeight: 600 }}>${(data.residentialSolar.economics.avgSystemCost / 1000).toFixed(0)}K</span>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', display: 'flex', justifyContent: 'space-between' }}>
                            <span>Cost Per Watt</span>
                            <span style={{ fontWeight: 600, color: 'var(--accent-green)' }}>${data.residentialSolar.economics.costPerWatt}</span>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', display: 'flex', justifyContent: 'space-between' }}>
                            <span>ROI Range</span>
                            <span style={{ fontWeight: 600, color: 'var(--accent-blue)' }}>{data.residentialSolar.economics.roi.min}-{data.residentialSolar.economics.roi.max}%</span>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', display: 'flex', justifyContent: 'space-between' }}>
                            <span>Home Value Increase</span>
                            <span style={{ fontWeight: 600, color: 'var(--accent-purple)' }}>+{data.residentialSolar.economics.homeValueIncrease}%</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Savings & ROI */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">💵 Solar Savings Over Time</h3>
                <div className="grid-3" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-green)' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-green)' }}>
                            ${data.residentialSolar.savings.annualSavings.toLocaleString()}
                        </div>
                        <div style={{ color: 'var(--text-secondary)' }}>Annual Savings</div>
                    </div>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-blue)' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-blue)' }}>
                            {data.residentialSolar.savings.electricityOffset.min}-{data.residentialSolar.savings.electricityOffset.max}%
                        </div>
                        <div style={{ color: 'var(--text-secondary)' }}>Bill Offset</div>
                    </div>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-purple)' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-purple)' }}>
                            ${(data.residentialSolar.savings.totalLifetimeSavings / 1000).toFixed(0)}K
                        </div>
                        <div style={{ color: 'var(--text-secondary)' }}>25-Year Savings</div>
                    </div>
                </div>
            </div>

            {/* Solar + EV Synergy */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🔌 Solar + EV: Energy Independence</h3>
                <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
                    {data.solarPlusEV.insight}
                </p>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-green)', marginBottom: '0.5rem' }}>Solar Production</h4>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                            {data.solarPlusEV.scenario.systemSize}kW system produces <strong>{(data.solarPlusEV.scenario.annualProduction / 1000).toFixed(0)}K kWh/year</strong>
                        </p>
                    </div>
                    <div className="card" style={{ padding: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-blue)', marginBottom: '0.5rem' }}>EV Consumption</h4>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                            {(data.solarPlusEV.scenario.evAnnualMiles / 1000).toFixed(0)}K miles/year needs only <strong>{(data.solarPlusEV.scenario.evAnnualKwh / 1000).toFixed(1)}K kWh</strong>
                        </p>
                    </div>
                </div>
                <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--bg-hover)', borderRadius: '8px', textAlign: 'center' }}>
                    <p style={{ color: 'var(--accent-green)', fontWeight: 600, fontSize: '1.1rem' }}>
                        ☀️ Solar covers {data.solarPlusEV.economics.solarCoverage}% of home + EV needs!
                    </p>
                </div>
            </div>

            {/* Top States */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🗺️ Top Solar States</h3>
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem', flexWrap: 'wrap' }}>
                    {data.residentialSolar.topStates.map((state, i) => (
                        <span key={i} className="badge badge-blue" style={{ padding: '0.5rem 1rem' }}>
                            {state}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    )
}
