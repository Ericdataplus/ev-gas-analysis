import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ScatterChart, Scatter, ZAxis } from 'recharts'
import ChartModal from '../components/ChartModal'
import mlInsights from '../data/ml_insights.json'

export default function MLInsights() {
    // Correlation data
    const correlationData = [
        { feature: 'Charging Stations', correlation: 0.996 },
        { feature: 'EV Market Share', correlation: 0.995 },
        { feature: 'Battery Density', correlation: 0.955 },
        { feature: 'EV Range', correlation: 0.920 },
        { feature: 'Battery Cost', correlation: -0.935 },
    ]

    // Feature importance
    const featureImportance = [
        { feature: 'Battery Density', rf: 0.314, mi: 0.547 },
        { feature: 'Charging Stations', rf: 0.289, mi: 0.814 },
        { feature: 'EV Range', rf: 0.238, mi: 0.437 },
        { feature: 'Battery Cost', rf: 0.158, mi: 0.069 },
    ]

    // CAGR data
    const cagrData = [
        { metric: 'EV Sales', cagr: 48.1, color: '#22c55e' },
        { metric: 'Charging Stations', cagr: 43.0, color: '#3b82f6' },
        { metric: 'Solar Capacity', cagr: 28.9, color: '#f97316' },
        { metric: 'Data Centers', cagr: 10.9, color: '#a855f7' },
        { metric: 'Trucking Revenue', cagr: 4.9, color: '#6b7280' },
    ]

    // Country clusters
    const countryData = [
        { country: 'Norway', ev: 89, solar: 300, gdp: 92, policy: 10, cluster: 'Leader' },
        { country: 'Sweden', ev: 58, solar: 350, gdp: 60, policy: 8, cluster: 'Leader' },
        { country: 'Netherlands', ev: 48, solar: 1337, gdp: 58, policy: 8, cluster: 'Leader' },
        { country: 'China', ev: 40, solar: 620, gdp: 12, policy: 9, cluster: 'Follower' },
        { country: 'Germany', ev: 19, solar: 1192, gdp: 51, policy: 7, cluster: 'Follower' },
        { country: 'UK', ev: 30, solar: 250, gdp: 46, policy: 7, cluster: 'Follower' },
        { country: 'USA', ev: 9, solar: 720, gdp: 76, policy: 5, cluster: 'Laggard' },
        { country: 'Australia', ev: 10, solar: 1400, gdp: 60, policy: 6, cluster: 'Laggard' },
    ]

    // Transport efficiency
    const transportData = [
        { mode: 'Ship', co2: 0.015, speed: 15, cost: 0.02 },
        { mode: 'Rail', co2: 0.025, speed: 35, cost: 0.03 },
        { mode: 'Truck', co2: 0.150, speed: 55, cost: 0.10 },
        { mode: 'Air', co2: 1.230, speed: 500, cost: 0.50 },
    ]

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🧠 Deep ML Insights</h1>
                <p className="page-subtitle">Non-predictive analysis: correlations, clusters, PCA, hypothesis tests</p>
            </header>

            {/* Key Findings Summary */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon">📊</div>
                    <div className="stat-value">0.996</div>
                    <div className="stat-label">Strongest Correlation</div>
                    <div className="stat-change">EV sales ↔ Charging</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🎯</div>
                    <div className="stat-value">p=0.001</div>
                    <div className="stat-label">Policy → EV Adoption</div>
                    <div className="stat-change">Highly significant!</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🚀</div>
                    <div className="stat-value">48.1%</div>
                    <div className="stat-label">EV Sales CAGR</div>
                    <div className="stat-change">Fastest growing sector</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🌍</div>
                    <div className="stat-value">3</div>
                    <div className="stat-label">Country Clusters</div>
                    <div className="stat-change">Leaders, Followers, Laggards</div>
                </div>
            </div>

            {/* Correlation Analysis */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📈 Correlation Analysis: What Moves Together?</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="EV Sales Correlations"
                        insight="Charging infrastructure has the strongest correlation with EV sales (r=0.996). Battery cost is strongly NEGATIVE (-0.935) - as costs drop, sales soar. This is textbook exponential adoption driven by price."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={correlationData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" domain={[-1, 1]} stroke="#71717a" />
                                <YAxis dataKey="feature" type="category" stroke="#71717a" width={120} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar dataKey="correlation" radius={[0, 8, 8, 0]}>
                                    {correlationData.map((entry, i) => (
                                        <Cell key={i} fill={entry.correlation > 0 ? '#22c55e' : '#ef4444'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Key Correlation Insights</h4>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.75rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <strong>🔗 Co-evolving factors:</strong>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                EV sales, charging stations, and range all grow in lockstep (r &gt; 0.95)
                            </p>
                        </div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.75rem', borderLeft: '3px solid var(--accent-red)' }}>
                            <strong>📉 Inverse relationship:</strong>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                Battery cost vs sales (r = -0.94). Price is THE barrier.
                            </p>
                        </div>
                        <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-purple)' }}>
                            <strong>💡 Insight:</strong>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                The EV transition is a self-reinforcing cycle: more sales → more charging → more adoption
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Feature Importance */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🎯 Feature Importance: What Drives EV Adoption?</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Random Forest vs Mutual Information"
                        insight="Interesting divergence: Random Forest says battery density matters most (0.31), but Mutual Information says charging stations (0.81). This suggests non-linear relationships - infrastructure has threshold effects!"
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={featureImportance}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="feature" stroke="#71717a" />
                                <YAxis stroke="#71717a" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar dataKey="rf" fill="#3b82f6" name="Random Forest" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="mi" fill="#a855f7" name="Mutual Info" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Feature Ranking</h4>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <span>1. Charging Stations</span>
                                <span style={{ color: 'var(--accent-purple)', fontWeight: 600 }}>MI: 0.81</span>
                            </div>
                            <div style={{ background: '#27272a', borderRadius: '4px', height: '8px' }}>
                                <div style={{ background: 'var(--accent-purple)', width: '81%', height: '100%', borderRadius: '4px' }}></div>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <span>2. Battery Density</span>
                                <span style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>MI: 0.55</span>
                            </div>
                            <div style={{ background: '#27272a', borderRadius: '4px', height: '8px' }}>
                                <div style={{ background: 'var(--accent-blue)', width: '55%', height: '100%', borderRadius: '4px' }}></div>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <span>3. EV Range</span>
                                <span style={{ color: 'var(--accent-orange)', fontWeight: 600 }}>MI: 0.44</span>
                            </div>
                            <div style={{ background: '#27272a', borderRadius: '4px', height: '8px' }}>
                                <div style={{ background: 'var(--accent-orange)', width: '44%', height: '100%', borderRadius: '4px' }}></div>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '1rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <span>4. Battery Cost</span>
                                <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>MI: 0.07</span>
                            </div>
                            <div style={{ background: '#27272a', borderRadius: '4px', height: '8px' }}>
                                <div style={{ background: 'var(--accent-green)', width: '7%', height: '100%', borderRadius: '4px' }}></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Growth Rates (CAGR) */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🚀 Compound Annual Growth Rates (CAGR)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Growth Rate Comparison"
                        insight="EV sales (48.1% CAGR) and charging stations (43.0%) are growing exponentially. Solar (28.9%) is also rocketing. Data centers (10.9%) growing 5x faster than overall electricity. Trucking (4.9%) is steady but mature."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={cagrData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" unit="%" />
                                <YAxis dataKey="metric" type="category" stroke="#71717a" width={120} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} formatter={(v) => `${v}%`} />
                                <Bar dataKey="cagr" radius={[0, 8, 8, 0]}>
                                    {cagrData.map((entry, i) => (
                                        <Cell key={i} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Doubling Times</h4>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>🚗 EV Sales</span>
                                <span style={{ fontWeight: 600 }}>Double every ~1.8 years</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>⚡ Charging Stations</span>
                                <span style={{ fontWeight: 600 }}>Double every ~2.0 years</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>☀️ Solar Capacity</span>
                                <span style={{ fontWeight: 600 }}>Double every ~2.7 years</span>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '1rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>🖥️ Data Centers</span>
                                <span style={{ fontWeight: 600 }}>Double every ~6.7 years</span>
                            </div>
                        </div>
                        <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'var(--bg-hover)', borderRadius: '8px', fontSize: '0.85rem' }}>
                            💡 Rule: 72 / CAGR% = Years to double
                        </div>
                    </div>
                </div>
            </div>

            {/* Country Clustering */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🌍 Country Clustering: Who's Leading the Transition?</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="EV Adoption by Policy Score"
                        insight="K-Means clustering reveals 3 distinct groups: Leaders (Norway, Sweden, Netherlands) with high policy + adoption; Followers (China, Germany, UK) with moderate; Laggards (USA, Australia) with low policy = low adoption. GDP doesn't matter - policy does!"
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="policy" name="Policy Score" stroke="#71717a" domain={[4, 11]} label={{ value: 'Policy Score', position: 'bottom' }} />
                                <YAxis dataKey="ev" name="EV Share %" stroke="#71717a" domain={[0, 100]} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Scatter data={countryData.filter(c => c.cluster === 'Leader')} fill="#22c55e" name="Leaders" />
                                <Scatter data={countryData.filter(c => c.cluster === 'Follower')} fill="#3b82f6" name="Followers" />
                                <Scatter data={countryData.filter(c => c.cluster === 'Laggard')} fill="#ef4444" name="Laggards" />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Cluster Breakdown</h4>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.75rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <h5 style={{ color: 'var(--accent-green)', margin: 0 }}>🏆 Leaders</h5>
                            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', margin: '0.25rem 0' }}>Norway, Sweden, Netherlands</p>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>Avg EV share: 65% | Avg Policy: 8.7</p>
                        </div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.75rem', borderLeft: '3px solid var(--accent-blue)' }}>
                            <h5 style={{ color: 'var(--accent-blue)', margin: 0 }}>📈 Followers</h5>
                            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', margin: '0.25rem 0' }}>China, Germany, UK</p>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>Avg EV share: 30% | Avg Policy: 7.7</p>
                        </div>
                        <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-red)' }}>
                            <h5 style={{ color: 'var(--accent-red)', margin: 0 }}>📉 Laggards</h5>
                            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', margin: '0.25rem 0' }}>USA, Australia</p>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>Avg EV share: 9.5% | Avg Policy: 5.5</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Hypothesis Tests */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🔬 Statistical Hypothesis Tests</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1.5rem' }}>
                        <h4 style={{ marginBottom: '1rem' }}>H1: GDP → EV Adoption?</h4>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                            <span>Spearman correlation:</span>
                            <span>r = 0.108</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                            <span>P-value:</span>
                            <span>p = 0.80</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                            <span>Significance (α=0.05):</span>
                            <span style={{ color: 'var(--accent-red)', fontWeight: 600 }}>NOT SIGNIFICANT ✗</span>
                        </div>
                        <div style={{ padding: '0.75rem', background: 'rgba(239, 68, 68, 0.2)', borderRadius: '8px' }}>
                            <p style={{ margin: 0, fontSize: '0.9rem' }}>
                                💡 <strong>Wealth alone does NOT drive EV adoption.</strong> USA is richer than Norway but has 10x fewer EVs.
                            </p>
                        </div>
                    </div>

                    <div className="card" style={{ padding: '1.5rem' }}>
                        <h4 style={{ marginBottom: '1rem' }}>H2: Policy → EV Adoption?</h4>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                            <span>Spearman correlation:</span>
                            <span>r = 0.916</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                            <span>P-value:</span>
                            <span>p = 0.001</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                            <span>Significance (α=0.05):</span>
                            <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>SIGNIFICANT ✓</span>
                        </div>
                        <div style={{ padding: '0.75rem', background: 'rgba(34, 197, 94, 0.2)', borderRadius: '8px' }}>
                            <p style={{ margin: 0, fontSize: '0.9rem' }}>
                                💡 <strong>Policy is THE key driver!</strong> Strong incentives (Norway) = high adoption. Weak policy (USA) = slow adoption.
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Bottom Line */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(168, 85, 247, 0.1))' }}>
                <h3 className="chart-title">🎯 Key ML Discoveries</h3>
                <div style={{ marginTop: '1rem', fontSize: '1.05rem', lineHeight: 1.8 }}>
                    <p><strong style={{ color: 'var(--accent-green)' }}>📊 CORRELATION:</strong> Battery cost inversely linked to sales (r=-0.94). Infrastructure co-evolves with adoption (r=0.99)</p>
                    <p><strong style={{ color: 'var(--accent-blue)' }}>🎯 FEATURE IMPORTANCE:</strong> Charging stations have highest mutual information (0.81) - threshold effects matter</p>
                    <p><strong style={{ color: 'var(--accent-purple)' }}>🌍 CLUSTERING:</strong> Countries form 3 clear groups by policy + adoption. GDP is NOT a good predictor</p>
                    <p><strong style={{ color: 'var(--accent-orange)' }}>📈 GROWTH:</strong> EV sales growing 48% CAGR - doubling every 1.8 years!</p>
                    <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(34, 197, 94, 0.2)', borderRadius: '8px', textAlign: 'center' }}>
                        <p style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: '1.1rem', margin: 0 }}>
                            🔬 The data is clear: Policy drives adoption, not wealth. The transition is exponential and accelerating.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
