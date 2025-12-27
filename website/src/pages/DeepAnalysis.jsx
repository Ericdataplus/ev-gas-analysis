import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ComposedChart, Line } from 'recharts'
import ChartModal from '../components/ChartModal'

export default function DeepAnalysis() {
    const gridData = [
        { year: 2024, evDemand: 120, renewable: 27 },
        { year: 2030, evDemand: 283, renewable: 40 },
        { year: 2040, evDemand: 600, renewable: 60 },
        { year: 2050, evDemand: 900, renewable: 80 },
    ]

    const batteryDensity = [
        { year: 2024, density: 280 },
        { year: 2030, density: 500 },
        { year: 2035, density: 650 },
        { year: 2040, density: 800 },
        { year: 2050, density: 1000 },
    ]

    const countryEV = [
        { country: 'Norway', share: 89 },
        { country: 'Sweden', share: 58 },
        { country: 'Netherlands', share: 48 },
        { country: 'China', share: 40 },
        { country: 'UK', share: 30 },
        { country: 'Germany', share: 19 },
        { country: 'Australia', share: 10 },
        { country: 'USA', share: 9 },
    ]

    const countrySolar = [
        { country: 'Australia', watts: 1400 },
        { country: 'Netherlands', watts: 1337 },
        { country: 'Germany', watts: 1192 },
        { country: 'USA', watts: 720 },
        { country: 'China', watts: 620 },
    ]

    const industrialData = [
        { sector: 'Steel', emissions: 2800, reduction: 90 },
        { sector: 'Cement', emissions: 2300, reduction: 44 },
        { sector: 'Shipping', emissions: 850, reduction: 100 },
    ]

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🔬 Deep Analysis</h1>
                <p className="page-subtitle">ML-powered predictions: Grid capacity, aviation, industry, and country rankings</p>
            </header>

            {/* Summary Cards */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon">⚡</div>
                    <div className="stat-value" style={{ color: 'var(--accent-green)' }}>YES</div>
                    <div className="stat-label">Grid Can Handle EVs</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🔋</div>
                    <div className="stat-value">4.1 yrs</div>
                    <div className="stat-label">Battery Payback 2030</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">✈️</div>
                    <div className="stat-value">2030</div>
                    <div className="stat-label">Regional Electric Flights</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🏆</div>
                    <div className="stat-value">Norway</div>
                    <div className="stat-label">EV Leader (89%)</div>
                </div>
            </div>

            {/* 1. GRID CAPACITY */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">⚡ Grid Capacity: Can It Handle All-EV Future?</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="EV Electricity Demand (TWh)"
                        insight="By 2050, EVs could require 900 TWh/year - but that's only ~18% of projected total demand. Smart charging during off-peak hours and V2G integration can easily handle this. The grid CAN support all-EV future with $2-4T infrastructure investment."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={gridData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis yAxisId="left" stroke="#71717a" />
                                <YAxis yAxisId="right" orientation="right" stroke="#71717a" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar yAxisId="left" dataKey="evDemand" fill="#3b82f6" radius={[4, 4, 0, 0]} name="EV Demand (TWh)" />
                                <Line yAxisId="right" type="monotone" dataKey="renewable" stroke="#22c55e" strokeWidth={2} name="Renewable %" />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </ChartModal>
                    <div>
                        <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.75rem' }}>Key Requirements</h4>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <strong>$2-4 trillion</strong> infrastructure investment
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-blue)' }}>
                            <strong>28M charging ports</strong> needed by 2030
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-purple)' }}>
                            <strong>Smart charging</strong> shifts 80% to off-peak
                        </div>
                        <div className="card" style={{ padding: '0.75rem', borderLeft: '3px solid var(--accent-orange)' }}>
                            <strong>V2G technology</strong> makes EVs grid assets
                        </div>
                        <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--bg-hover)', borderRadius: '8px' }}>
                            <p style={{ color: 'var(--accent-green)', fontWeight: 600, margin: 0 }}>
                                ✅ Verdict: Grid CAN handle all-EV future
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* 2. HOME BATTERIES */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🔋 Home Batteries: Worth the Investment?</h3>
                <div className="grid-3" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-orange)' }}>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>2024 Payback</div>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-orange)' }}>8.3 yrs</div>
                    </div>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-green)' }}>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>2030 Payback (with Solar)</div>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-green)' }}>4.1 yrs</div>
                    </div>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-blue)' }}>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Annual Savings</div>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-blue)' }}>$1,800</div>
                    </div>
                </div>
                <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--bg-hover)', borderRadius: '8px' }}>
                    <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
                        <strong style={{ color: 'var(--accent-green)' }}>✅ Worth it with solar + TOU rates.</strong> Tesla Powerwall has 47% market share.
                        Best ROI in high-rate states like CA, HI. Battery costs dropping 40% by 2030.
                    </p>
                </div>
            </div>

            {/* 3. ELECTRIC AVIATION */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">✈️ Electric Aviation: Is It Possible?</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Battery Energy Density (Wh/kg)"
                        insight="Jet fuel has 43 MJ/kg, current batteries only 1 MJ/kg (250 Wh/kg). For regional flights, we need 500 Wh/kg (achievable by 2030). For intercontinental, we need 1000+ Wh/kg (not until 2050+). Short-haul electric aviation is happening, long-haul won't."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={batteryDensity}>
                                <defs>
                                    <linearGradient id="densityGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Area type="monotone" dataKey="density" stroke="#a855f7" fill="url(#densityGrad)" strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </ChartModal>
                    <div>
                        <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.75rem' }}>Feasibility Timeline</h4>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Short-haul (&lt;500km)</span>
                                <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>2030 ✓</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Medium-haul (50-100 pax)</span>
                                <span style={{ color: 'var(--accent-orange)', fontWeight: 600 }}>2035-2040</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Long-haul intercontinental</span>
                                <span style={{ color: 'var(--accent-red)', fontWeight: 600 }}>Post-2050</span>
                            </div>
                        </div>
                        <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--bg-hover)', borderRadius: '8px' }}>
                            <p style={{ color: 'var(--accent-orange)', fontWeight: 600, margin: 0 }}>
                                ⚠️ Regional by 2030, long-haul unlikely before 2050
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* 4. INDUSTRIAL DECARBONIZATION */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🏭 Industrial Decarbonization: Steel, Cement, Shipping</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Current Emissions (Mt CO2/year)"
                        insight="Steel (8% of global emissions) can be decarbonized 90% with green hydrogen. Cement (8%) is harder - only 44% reduction with H2, needs carbon capture for rest. Shipping can go 100% green with e-methanol/e-ammonia by 2035."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={industrialData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" />
                                <YAxis dataKey="sector" type="category" stroke="#71717a" width={80} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar dataKey="emissions" fill="#ef4444" radius={[0, 8, 8, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>
                    <div>
                        <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.75rem' }}>H2 Reduction Potential</h4>
                        {industrialData.map((item, i) => (
                            <div key={i} className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                    <span>{item.sector}</span>
                                    <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>{item.reduction}%</span>
                                </div>
                                <div style={{ background: '#27272a', borderRadius: '4px', height: '8px' }}>
                                    <div style={{
                                        background: 'var(--accent-green)',
                                        width: `${item.reduction}%`,
                                        height: '100%',
                                        borderRadius: '4px'
                                    }}></div>
                                </div>
                            </div>
                        ))}
                        <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--bg-hover)', borderRadius: '8px' }}>
                            <p style={{ color: 'var(--accent-green)', fontWeight: 600, margin: 0 }}>
                                ✅ Achievable with $100B+ green hydrogen investment
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* 5. COUNTRY RANKINGS */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🌍 Who's Winning the Clean Energy Race?</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="EV Market Share by Country (%)"
                        insight="Norway leads with 89% EV share - nearly 9 in 10 new cars are electric! On track for 100% by 2025. China leads in volume (11M EVs in 2024). USA lags at just 9% but growing."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={countryEV} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" domain={[0, 100]} />
                                <YAxis dataKey="country" type="category" stroke="#71717a" width={80} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar dataKey="share" radius={[0, 8, 8, 0]}>
                                    {countryEV.map((entry, i) => (
                                        <Cell key={i} fill={entry.share > 50 ? '#22c55e' : entry.share > 25 ? '#3b82f6' : '#f97316'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>
                    <ChartModal
                        title="Solar Per Capita (Watts)"
                        insight="Australia leads globally with 1,400 watts per person - enough to power a small appliance 24/7 from solar alone! Netherlands and Germany follow. Despite massive installations, China's per-capita is lower due to population size."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={countrySolar} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" />
                                <YAxis dataKey="country" type="category" stroke="#71717a" width={100} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar dataKey="watts" fill="#eab308" radius={[0, 8, 8, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>
                </div>
            </div>

            {/* Final Verdict */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1))' }}>
                <h3 className="chart-title">🎯 Bottom Line: Can We Transition to Clean Energy?</h3>
                <div style={{ marginTop: '1rem', fontSize: '1.1rem', lineHeight: 1.8 }}>
                    <p><strong style={{ color: 'var(--accent-green)' }}>✅ GRID:</strong> Yes, can handle all-EVs with smart charging + $2-4T infrastructure</p>
                    <p><strong style={{ color: 'var(--accent-green)' }}>✅ HOME BATTERIES:</strong> Worth it with solar (4-year payback by 2030)</p>
                    <p><strong style={{ color: 'var(--accent-orange)' }}>⚠️ AVIATION:</strong> Regional by 2030, long-haul post-2050</p>
                    <p><strong style={{ color: 'var(--accent-green)' }}>✅ INDUSTRY:</strong> Steel/shipping solvable with green H2 by 2040</p>
                    <p><strong style={{ color: 'var(--accent-blue)' }}>🏆 LEADERS:</strong> Norway (EVs), Australia (Solar), China (Volume)</p>
                    <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(34, 197, 94, 0.2)', borderRadius: '8px', textAlign: 'center' }}>
                        <p style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: '1.25rem', margin: 0 }}>
                            The transition is technically feasible and economically improving every year.
                            The main barriers are investment speed and policy.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
