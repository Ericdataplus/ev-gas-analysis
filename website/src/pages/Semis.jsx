import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ComposedChart, PieChart, Pie } from 'recharts'
import ChartModal from '../components/ChartModal'

export default function Semis() {
    // Historical fleet size data
    const fleetHistory = [
        { year: 1990, trucks: 6.2, tractorTrailers: 1.1 },
        { year: 1995, trucks: 7.1, tractorTrailers: 1.4 },
        { year: 2000, trucks: 8.0, tractorTrailers: 1.7 },
        { year: 2005, trucks: 9.5, tractorTrailers: 2.2 },
        { year: 2010, trucks: 11.0, tractorTrailers: 2.4 },
        { year: 2015, trucks: 12.5, tractorTrailers: 2.7 },
        { year: 2020, trucks: 13.5, tractorTrailers: 2.9 },
        { year: 2024, trucks: 14.9, tractorTrailers: 3.0 },
    ]

    // Freight tonnage moved by trucks
    const freightHistory = [
        { year: 1990, tons: 6.5, pctOfFreight: 55 },
        { year: 1995, tons: 7.5, pctOfFreight: 57 },
        { year: 2000, tons: 8.5, pctOfFreight: 59 },
        { year: 2005, tons: 9.5, pctOfFreight: 65 },
        { year: 2010, tons: 9.0, pctOfFreight: 68 },
        { year: 2015, tons: 10.1, pctOfFreight: 70 },
        { year: 2020, tons: 10.2, pctOfFreight: 71 },
        { year: 2024, tons: 11.3, pctOfFreight: 73 },
    ]

    // Diesel fuel consumption (billions of gallons)
    const fuelHistory = [
        { year: 1990, gallons: 16.1 },
        { year: 1995, gallons: 20.5 },
        { year: 2000, gallons: 25.7 },
        { year: 2005, gallons: 28.5 },
        { year: 2010, gallons: 29.9 },
        { year: 2015, gallons: 35.2 },
        { year: 2020, gallons: 38.0 },
        { year: 2024, gallons: 42.0 },
    ]

    // Industry revenue (billions)
    const revenueHistory = [
        { year: 1990, revenue: 180 },
        { year: 2000, revenue: 206 },
        { year: 2010, revenue: 544 },
        { year: 2015, revenue: 650 },
        { year: 2020, revenue: 732 },
        { year: 2023, revenue: 1004 },
        { year: 2024, revenue: 906 },
    ]

    // Current industry breakdown
    const industryBreakdown = [
        { name: 'Class 8 (Semis)', value: 3.0, color: '#ef4444' },
        { name: 'Class 6-7 (Medium)', value: 3.5, color: '#f97316' },
        { name: 'Class 3-5 (Light)', value: 8.4, color: '#eab308' },
    ]

    const fuelTypeBreakdown = [
        { name: 'Diesel', value: 97.5, color: '#ef4444' },
        { name: 'Natural Gas', value: 1.5, color: '#3b82f6' },
        { name: 'Electric', value: 0.1, color: '#22c55e' },
        { name: 'Other', value: 0.9, color: '#6b7280' },
    ]

    // Electric semi market projection
    const electricProjection = [
        { year: 2024, electric: 0.1, diesel: 99.9 },
        { year: 2026, electric: 0.5, diesel: 99.5 },
        { year: 2028, electric: 2.0, diesel: 98.0 },
        { year: 2030, electric: 5.0, diesel: 95.0 },
        { year: 2035, electric: 15.0, diesel: 85.0 },
        { year: 2040, electric: 30.0, diesel: 70.0 },
    ]

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🚛 Semi Trucks & Freight Industry</h1>
                <p className="page-subtitle">America's supply chain backbone: 73% of all freight, $906B industry</p>
            </header>

            {/* Critical Stats */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon">🚛</div>
                    <div className="stat-value">14.9M</div>
                    <div className="stat-label">Commercial Trucks</div>
                    <div className="stat-change">3M are Class 8 semis</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">📦</div>
                    <div className="stat-value">73%</div>
                    <div className="stat-label">Freight by Weight</div>
                    <div className="stat-change">11.3B tons in 2024</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">💰</div>
                    <div className="stat-value">$906B</div>
                    <div className="stat-label">Industry Revenue</div>
                    <div className="stat-change">-10% from 2023</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">👷</div>
                    <div className="stat-value">3.6M</div>
                    <div className="stat-label">Truck Drivers</div>
                    <div className="stat-change">78K shortage</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">⛽</div>
                    <div className="stat-value">42B</div>
                    <div className="stat-label">Gallons/Year</div>
                    <div className="stat-change">~97% diesel</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🌡️</div>
                    <div className="stat-value">426M</div>
                    <div className="stat-label">Tons CO2/Year</div>
                    <div className="stat-change">~24% of transport</div>
                </div>
            </div>

            {/* Historical Fleet Growth */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📈 Historical Growth: Fleet Size (1990-2024)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Commercial Trucks (Millions)"
                        insight="The US truck fleet grew 140% from 6.2M in 1990 to 14.9M in 2024. Class 8 tractor-trailers (semis) grew from 1.1M to 3.0M. This growth tracks closely with GDP and consumer spending, as trucks are the lifeblood of commerce."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={fleetHistory}>
                                <defs>
                                    <linearGradient id="fleetGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" unit="M" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Area type="monotone" dataKey="trucks" stroke="#ef4444" fill="url(#fleetGrad)" strokeWidth={2} name="Total Trucks" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <ChartModal
                        title="Freight Tonnage Moved (Billions of Tons)"
                        insight="Trucks moved 6.5B tons in 1990, now move 11.3B tons - a 74% increase. More importantly, trucking's SHARE of freight grew from 55% to 73%. Rail and shipping declined as just-in-time delivery and e-commerce favored trucks' flexibility."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={freightHistory}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis yAxisId="left" stroke="#71717a" />
                                <YAxis yAxisId="right" orientation="right" stroke="#71717a" unit="%" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar yAxisId="left" dataKey="tons" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Billions of Tons" />
                                <Line yAxisId="right" type="monotone" dataKey="pctOfFreight" stroke="#22c55e" strokeWidth={2} name="% of All Freight" />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </ChartModal>
                </div>
            </div>

            {/* Fuel & Emissions */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">⛽ Diesel Consumption & Emissions</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Annual Diesel Consumption (Billions of Gallons)"
                        insight="Truck diesel consumption grew from 16.1B gallons in 1990 to 42B gallons in 2024 - a 161% increase. This is despite fuel efficiency improvements (5 MPG → 7 MPG). Volume growth outpaced efficiency gains."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={fuelHistory}>
                                <defs>
                                    <linearGradient id="fuelGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" unit="B" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Area type="monotone" dataKey="gallons" stroke="#f97316" fill="url(#fuelGrad)" strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.75rem' }}>Environmental Impact</h4>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-red)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>CO2 per gallon diesel</span>
                                <span style={{ fontWeight: 600 }}>22.4 lbs</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-red)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Per truck per year</span>
                                <span style={{ fontWeight: 600 }}>~126 tons CO2</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-red)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Industry total/year</span>
                                <span style={{ fontWeight: 600 }}>~426M tons CO2</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '0.75rem', borderLeft: '3px solid var(--accent-orange)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>% of US transport emissions</span>
                                <span style={{ fontWeight: 600 }}>~24%</span>
                            </div>
                        </div>
                        <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'var(--bg-hover)', borderRadius: '8px' }}>
                            <p style={{ fontSize: '0.9rem', margin: 0, color: 'var(--text-muted)' }}>
                                Heavy-duty trucks are the biggest challenge for transport decarbonization: high energy needs + long distances
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Industry Economics */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">💰 Industry Economics (1990-2024)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Industry Revenue (Billions $)"
                        insight="Trucking revenue grew from $180B in 1990 to over $1 trillion in 2023 - a 5x increase! The 2024 drop to $906B reflects freight recession and over-capacity, but structurally the industry only grows."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={revenueHistory}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" unit="B" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} formatter={(v) => `$${v}B`} />
                                <Bar dataKey="revenue" fill="#22c55e" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.75rem' }}>Why Trucks Dominate</h4>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                            <strong style={{ color: 'var(--accent-green)' }}>Flexibility:</strong> Door-to-door delivery, no transfer
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                            <strong style={{ color: 'var(--accent-blue)' }}>Speed:</strong> Faster than rail/ship for most routes
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                            <strong style={{ color: 'var(--accent-purple)' }}>Just-in-Time:</strong> Essential for modern supply chains
                        </div>
                        <div className="card" style={{ padding: '0.75rem' }}>
                            <strong style={{ color: 'var(--accent-orange)' }}>E-commerce:</strong> Last-mile delivery explosion
                        </div>
                    </div>
                </div>
            </div>

            {/* Driver Shortage */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">👷 The Driver Shortage Crisis</h3>
                <div className="grid-3" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-red)' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-red)' }}>78,000</div>
                        <div style={{ color: 'var(--text-secondary)' }}>Current Shortage</div>
                    </div>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-orange)' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-orange)' }}>1.2M</div>
                        <div style={{ color: 'var(--text-secondary)' }}>Needed Next Decade</div>
                    </div>
                    <div className="card" style={{ padding: '1.5rem', textAlign: 'center', borderLeft: '3px solid var(--accent-green)' }}>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-green)' }}>$57,440</div>
                        <div style={{ color: 'var(--text-secondary)' }}>Median Annual Wage</div>
                    </div>
                </div>
                <div style={{ marginTop: '1rem' }}>
                    <h4 style={{ marginBottom: '0.5rem' }}>Why the Shortage?</h4>
                    <div className="grid-2">
                        <div className="card" style={{ padding: '0.75rem' }}>😔 Long hours away from home</div>
                        <div className="card" style={{ padding: '0.75rem' }}>👴 Aging workforce (avg age: 48)</div>
                        <div className="card" style={{ padding: '0.75rem' }}>📋 Strict regulations (FMCSA)</div>
                        <div className="card" style={{ padding: '0.75rem' }}>💊 Drug testing sidelines many</div>
                    </div>
                </div>
            </div>

            {/* Current Power Mix & Electric Future */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">⚡ Power Mix: Today vs Future</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Current Fuel Type (% of Fleet)"
                        insight="97.5% of trucks still run on diesel. Electric is only 0.1% despite the hype. Natural gas is 1.5%. The transition will be slow due to: 1) High costs 2) Range needs 3) Weight limits 4) Charging infrastructure."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie data={fuelTypeBreakdown} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name}: ${value}%`}>
                                    {fuelTypeBreakdown.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <ChartModal
                        title="Electric Semi Projection (% of New Sales)"
                        insight="Electric semis will grow from 0.1% to maybe 30% by 2040. The main players are Tesla (500-mile range) and Nikola (hydrogen). But for long-haul (500+ miles), diesel/hydrogen will likely persist past 2050."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={electricProjection}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" unit="%" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Area type="monotone" dataKey="electric" stroke="#22c55e" fill="#22c55e" fillOpacity={0.3} strokeWidth={2} name="Electric %" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </ChartModal>
                </div>
            </div>

            {/* Key Questions */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">❓ Key Questions About Our Truck Dependence</h3>
                <div style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-blue)', marginBottom: '0.5rem' }}>What if trucking stopped for 1 week?</h4>
                        <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.9rem' }}>
                            Grocery stores empty in 3 days, gas stations in 2 days. Hospitals run out of supplies. ATMs run dry.
                            73% of all goods would stop moving. The economy loses ~$1.7B/day.
                        </p>
                    </div>
                    <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-green)', marginBottom: '0.5rem' }}>Can we shift freight to rail?</h4>
                        <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.9rem' }}>
                            Partially. Rail is 3-4x more fuel efficient but lacks flexibility. Only works for non-time-sensitive bulk goods.
                            Rail share has actually DECLINED (from 38% to 27%) as just-in-time inventory became dominant.
                        </p>
                    </div>
                    <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-purple)', marginBottom: '0.5rem' }}>Will autonomous trucks solve the driver shortage?</h4>
                        <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.9rem' }}>
                            Eventually. Companies like Aurora, Waymo, and TuSimple are testing. But regulatory approval, safety concerns,
                            and last-mile needs mean human drivers will be needed for decades. Expect highway-only autonomy first by 2030.
                        </p>
                    </div>
                    <div className="card" style={{ padding: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-orange)', marginBottom: '0.5rem' }}>Can electric semis replace diesel?</h4>
                        <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.9rem' }}>
                            For regional routes (&lt;300 miles): Yes, by 2030-2035. For long-haul (500+ miles): Unlikely before 2050.
                            Battery weight eats into payload. Hydrogen fuel cells may be better for long-haul. Tesla Semi shows promise but
                            only delivered 200 units so far vs 3M diesel semis on the road.
                        </p>
                    </div>
                </div>
            </div>

            {/* Bottom Line */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(251, 146, 60, 0.1))' }}>
                <h3 className="chart-title">🎯 Bottom Line: How Dependent Are We?</h3>
                <div style={{ marginTop: '1rem', fontSize: '1.05rem', lineHeight: 1.8 }}>
                    <p><strong>EXTREMELY.</strong> The trucking industry is irreplaceable for modern life:</p>
                    <ul style={{ color: 'var(--text-secondary)', marginLeft: '1.5rem' }}>
                        <li>73% of all freight moves by truck (11.3B tons/year)</li>
                        <li>$906B industry employing 8.4M people</li>
                        <li>Everything you buy touched a truck at some point</li>
                        <li>E-commerce makes us MORE dependent, not less</li>
                    </ul>
                    <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.2)', borderRadius: '8px' }}>
                        <p style={{ fontWeight: 600, margin: 0 }}>
                            ⚠️ Decarbonizing trucking is one of the hardest climate challenges. 97.5% run on diesel today.
                            Electric works for regional, but long-haul will need hydrogen or synthetic fuels.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
