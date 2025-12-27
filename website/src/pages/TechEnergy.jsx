import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ComposedChart, PieChart, Pie } from 'recharts'
import ChartModal from '../components/ChartModal'

export default function TechEnergy() {
    // Data center consumption over time
    const datacenterGrowth = [
        { year: 2015, twh: 200 },
        { year: 2018, twh: 280 },
        { year: 2020, twh: 300 },
        { year: 2022, twh: 350 },
        { year: 2024, twh: 415 },
        { year: 2026, twh: 600 },
        { year: 2028, twh: 750 },
        { year: 2030, twh: 945 },
    ]

    // AI portion of data center energy
    const aiGrowth = [
        { year: 2022, ai: 20, other: 330 },
        { year: 2024, ai: 60, other: 355 },
        { year: 2026, ai: 150, other: 450 },
        { year: 2028, ai: 300, other: 450 },
        { year: 2030, ai: 500, other: 445 },
    ]

    // AI model training energy comparison
    const modelTraining = [
        { model: 'BERT', mwh: 0.3, co2: 0.6 },
        { model: 'GPT-2', mwh: 280, co2: 140 },
        { model: 'GPT-3', mwh: 1287, co2: 552 },
        { model: 'GPT-4', mwh: 3500, co2: 7138 },
        { model: 'GPT-5 (est)', mwh: 7000, co2: 15000 },
    ]

    // Digital activity CO2 comparison
    const digitalCO2 = [
        { activity: 'Email (no attach)', grams: 4 },
        { activity: 'Google search', grams: 0.2 },
        { activity: 'ChatGPT query', grams: 4.7 },
        { activity: '1hr Netflix (phone)', grams: 0.5 },
        { activity: '1hr Netflix (TV)', grams: 55 },
        { activity: 'Email (10MB attach)', grams: 50 },
    ]

    // Tech company energy use
    const techCompanyEnergy = [
        { company: 'Google', twh: 25.3, renewable: 100 },
        { company: 'Microsoft', twh: 23.0, renewable: 100 },
        { company: 'Amazon (AWS)', twh: 20.0, renewable: 90 },
        { company: 'Meta', twh: 12.5, renewable: 100 },
        { company: 'Apple', twh: 4.5, renewable: 100 },
    ]

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">💻 Tech & AI Energy Consumption</h1>
                <p className="page-subtitle">Data centers, AI training, streaming, and the digital carbon footprint</p>
            </header>

            {/* Key Stats */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon">🏢</div>
                    <div className="stat-value">415 TWh</div>
                    <div className="stat-label">Data Centers (2024)</div>
                    <div className="stat-change">1.5% of global electricity</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🤖</div>
                    <div className="stat-value">10x</div>
                    <div className="stat-label">ChatGPT vs Google</div>
                    <div className="stat-change">Energy per query</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">📈</div>
                    <div className="stat-value">945 TWh</div>
                    <div className="stat-label">Projected 2030</div>
                    <div className="stat-change">+128% from 2024</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🎬</div>
                    <div className="stat-value">55g</div>
                    <div className="stat-label">CO2/hr Streaming (TV)</div>
                    <div className="stat-change">0.5g on phone</div>
                </div>
            </div>

            {/* Data Center Growth */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🏢 Global Data Center Energy Consumption (TWh)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Data Center Electricity Growth"
                        insight="Data centers consumed 415 TWh in 2024 (1.5% of global electricity). By 2030, this could reach 945 TWh - more than doubling! AI is driving most of this growth, with computational demands doubling every 100 days."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={datacenterGrowth}>
                                <defs>
                                    <linearGradient id="dcGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} formatter={(v) => `${v} TWh`} />
                                <Area type="monotone" dataKey="twh" stroke="#3b82f6" fill="url(#dcGrad)" strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Key Drivers</h4>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-purple)' }}>
                            <strong>🤖 AI/ML Workloads:</strong> Fastest growing segment
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-blue)' }}>
                            <strong>☁️ Cloud Services:</strong> AWS, Azure, GCP expansion
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', borderLeft: '3px solid var(--accent-orange)' }}>
                            <strong>🎮 Gaming/Streaming:</strong> 4K/8K video, cloud gaming
                        </div>
                        <div className="card" style={{ padding: '0.75rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <strong>💰 Crypto Mining:</strong> Bitcoin alone ~150 TWh/yr
                        </div>
                    </div>
                </div>
            </div>

            {/* AI vs Other Data Center Load */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🤖 AI's Growing Share of Data Center Energy</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="AI vs Traditional Data Center Load (TWh)"
                        insight="In 2024, AI accounts for ~15% of data center energy. By 2030, AI could consume 500+ TWh - over half of all data center electricity! This is why tech companies are racing to build nuclear plants and solar farms."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={aiGrowth}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar dataKey="other" stackId="a" fill="#6b7280" name="Traditional" />
                                <Bar dataKey="ai" stackId="a" fill="#a855f7" name="AI Workloads" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>AI Energy Facts</h4>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.75rem' }}>
                            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-purple)' }}>100 days</div>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>AI compute demand doubles every 100 days</div>
                        </div>
                        <div className="card" style={{ padding: '1rem' }}>
                            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-orange)' }}>15% → 50%+</div>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>AI share of data center energy by 2030</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* AI Model Training Comparison */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">⚡ AI Model Training Energy (MWh)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Training Energy by Model"
                        insight="GPT-4 training consumed ~3,500 MWh - equivalent to 350 US homes for a year! Each new generation requires 2-3x more energy. GPT-5 may need 7,000+ MWh. This is why NVIDIA GPUs are selling like goldmines."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={modelTraining} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" />
                                <YAxis dataKey="model" type="category" stroke="#71717a" width={80} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} formatter={(v) => `${v} MWh`} />
                                <Bar dataKey="mwh" fill="#ef4444" radius={[0, 8, 8, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>CO2 Emissions (tons)</h4>
                        {modelTraining.map((m, i) => (
                            <div key={i} className="card" style={{ padding: '0.5rem 0.75rem', marginBottom: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <span>{m.model}</span>
                                    <span style={{ color: m.co2 > 1000 ? 'var(--accent-red)' : 'var(--text-secondary)', fontWeight: 600 }}>{m.co2.toLocaleString()} tons</span>
                                </div>
                            </div>
                        ))}
                        <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: 'var(--bg-hover)', borderRadius: '8px', fontSize: '0.85rem' }}>
                            💡 GPT-4's CO2 = 1,550 Americans' annual emissions
                        </div>
                    </div>
                </div>
            </div>

            {/* Digital Activity CO2 */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🌍 Carbon Footprint of Digital Activities (grams CO2)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="CO2 per Activity"
                        insight="A ChatGPT query uses ~10x more energy than a Google search (4.7g vs 0.2g CO2). Streaming on a TV produces 100x more CO2 than on a phone due to the screen. Emails with attachments are surprisingly carbon-heavy!"
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={digitalCO2} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" />
                                <YAxis dataKey="activity" type="category" stroke="#71717a" width={130} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} formatter={(v) => `${v}g CO2`} />
                                <Bar dataKey="grams" radius={[0, 8, 8, 0]}>
                                    {digitalCO2.map((entry, i) => (
                                        <Cell key={i} fill={entry.grams > 10 ? '#ef4444' : entry.grams > 1 ? '#f97316' : '#22c55e'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Per-Query Energy</h4>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.75rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span>🔍 Google Search</span>
                                <span style={{ fontWeight: 700 }}>0.0003 kWh</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.75rem', borderLeft: '3px solid var(--accent-orange)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span>🤖 ChatGPT Query</span>
                                <span style={{ fontWeight: 700 }}>0.003 kWh</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-purple)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span>🖼️ AI Image Gen</span>
                                <span style={{ fontWeight: 700 }}>0.02 kWh</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Tech Companies */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🏢 Big Tech Energy Consumption (TWh/year)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Tech Giant Energy Use"
                        insight="Google, Microsoft, and AWS each consume 20-25 TWh/year - more than many small countries! The good news: all major tech companies have committed to 100% renewable energy. They're now building their own solar farms and considering nuclear."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={techCompanyEnergy}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="company" stroke="#71717a" />
                                <YAxis stroke="#71717a" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} formatter={(v) => `${v} TWh/yr`} />
                                <Bar dataKey="twh" radius={[4, 4, 0, 0]}>
                                    {techCompanyEnergy.map((entry, i) => (
                                        <Cell key={i} fill={['#4285f4', '#00a4ef', '#ff9900', '#0668E1', '#555555'][i]} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Renewable Energy %</h4>
                        {techCompanyEnergy.map((c, i) => (
                            <div key={i} className="card" style={{ padding: '0.5rem 0.75rem', marginBottom: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                    <span>{c.company}</span>
                                    <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>{c.renewable}%</span>
                                </div>
                                <div style={{ background: '#27272a', borderRadius: '4px', height: '6px' }}>
                                    <div style={{ background: 'var(--accent-green)', width: `${c.renewable}%`, height: '100%', borderRadius: '4px' }}></div>
                                </div>
                            </div>
                        ))}
                        <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: 'rgba(34, 197, 94, 0.2)', borderRadius: '8px', fontSize: '0.85rem' }}>
                            ✅ All major tech companies at 90-100% renewable!
                        </div>
                    </div>
                </div>
            </div>

            {/* Key Questions */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">❓ Key Questions About Tech Energy</h3>
                <div style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-purple)', marginBottom: '0.5rem' }}>Is AI growth sustainable?</h4>
                        <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.9rem' }}>
                            Short-term: Concerning. AI compute doubles every 100 days. Long-term: Tech companies are investing heavily in
                            renewables and nuclear. Microsoft, Google, Amazon are all building dedicated clean energy for data centers.
                        </p>
                    </div>
                    <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-blue)', marginBottom: '0.5rem' }}>Should I feel guilty about streaming?</h4>
                        <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.9rem' }}>
                            Not really. 1 hour of Netflix = 55g CO2 (on TV), which equals driving 0.2 miles in a gas car. Your TV is the
                            bigger factor - streaming on phone uses 100x less energy. The device matters more than the streaming.
                        </p>
                    </div>
                    <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-orange)', marginBottom: '0.5rem' }}>Is ChatGPT worse than Google?</h4>
                        <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.9rem' }}>
                            Yes, ~10x more energy per query. But context matters: if ChatGPT saves you 10 Google searches by giving a
                            better answer, it's a wash. AI is also getting more efficient - GPT-4o uses 10x less energy than GPT-4.
                        </p>
                    </div>
                    <div className="card" style={{ padding: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-green)', marginBottom: '0.5rem' }}>What's being done about it?</h4>
                        <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.9rem' }}>
                            • All major tech at 90-100% renewable energy<br />
                            • Microsoft investing in nuclear fusion<br />
                            • Google buying 24/7 carbon-free energy<br />
                            • New AI chips (H100) are 2-3x more efficient<br />
                            • Data centers moving to cooler climates (Sweden, Norway)
                        </p>
                    </div>
                </div>
            </div>

            {/* Bottom Line */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(168, 85, 247, 0.1))' }}>
                <h3 className="chart-title">🎯 Bottom Line: Is Tech Destroying the Planet?</h3>
                <div style={{ marginTop: '1rem', fontSize: '1.05rem', lineHeight: 1.8 }}>
                    <p><strong style={{ color: 'var(--accent-blue)' }}>⚠️ THE CONCERN:</strong> Data centers use 415 TWh/yr (1.5% of global electricity), doubling by 2030</p>
                    <p><strong style={{ color: 'var(--accent-purple)' }}>🤖 AI IS THE DRIVER:</strong> AI compute doubles every 100 days. GPT-4 training = 3,500 MWh</p>
                    <p><strong style={{ color: 'var(--accent-green)' }}>✅ THE GOOD NEWS:</strong> Big Tech is 90-100% renewable and investing heavily in clean energy</p>
                    <p><strong style={{ color: 'var(--accent-orange)' }}>📊 PERSPECTIVE:</strong> All data centers = ~1.5% of electricity. All cars = ~16% of emissions.</p>
                    <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(34, 197, 94, 0.2)', borderRadius: '8px', textAlign: 'center' }}>
                        <p style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: '1.1rem', margin: 0 }}>
                            Tech's energy use is growing fast, but it's also leading the transition to clean energy.
                            The bigger climate wins are in transport, heating, and industry.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
