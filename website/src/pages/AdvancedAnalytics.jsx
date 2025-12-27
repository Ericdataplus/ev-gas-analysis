import { useState } from 'react'
import {
    AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis,
    CartesianGrid, Tooltip, ResponsiveContainer, Cell, RadarChart,
    PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ScatterChart,
    Scatter, ComposedChart, Legend, ReferenceLine
} from 'recharts'

// Import the new analysis data
import advancedData from '../data/advanced_ml_analysis.json'
import ultraDeepData from '../data/ultra_deep_ml_analysis.json'

const COLORS = {
    primary: '#6366f1',
    secondary: '#8b5cf6',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    info: '#06b6d4',
    pink: '#ec4899',
    lime: '#84cc16'
}

export default function AdvancedAnalytics() {
    const [activeSection, setActiveSection] = useState('monte-carlo')

    // Monte Carlo data
    const monteCarloData = ultraDeepData?.monte_carlo || {}
    const batteryMC = monteCarloData.battery_cost_mc || {}
    const evSalesMC = monteCarloData.ev_sales_mc || {}

    // Granger Causality data
    const causalityData = advancedData?.causality?.causality_tests || []

    // Seasonality data
    const seasonalityData = ultraDeepData?.seasonality?.seasonality_patterns || {}

    // VaR data
    const varData = ultraDeepData?.var_analysis?.var_analysis || {}

    // Stress testing data
    const stressData = ultraDeepData?.stress_testing || {}

    // Clustering data
    const clusterData = advancedData?.clustering || {}

    // Anomaly data
    const anomalyData = advancedData?.anomalies || {}

    // Regime data
    const regimeData = advancedData?.regimes?.regimes || {}

    // Format VaR for chart
    const varChartData = Object.entries(varData)
        .map(([name, data]) => ({
            name: name.length > 12 ? name.substring(0, 12) + '...' : name,
            fullName: name,
            var95: Math.abs(data.daily_var_95),
            volatility: data.annualized_volatility,
            worstDay: Math.abs(data.worst_day_pct)
        }))
        .sort((a, b) => b.var95 - a.var95)
        .slice(0, 12)

    // Format causality for chart
    const causalityChartData = causalityData.map(item => ({
        relationship: `${item.cause.substring(0, 10)}→${item.effect.substring(0, 10)}`,
        fullRelationship: `${item.cause} → ${item.effect}`,
        pValue: -Math.log10(item.p_value + 0.0001),
        lag: item.best_lag_months,
        significant: item.significant === 'True' || item.significant === true
    }))

    // Format seasonality for chart
    const gasSeasonality = seasonalityData.gas_price_regular?.monthly_factors || {}
    const seasonalityChartData = Object.entries(gasSeasonality).map(([month, factor]) => ({
        month,
        factor: factor - 100,
        value: factor
    }))

    // Format stress scenarios
    const stressChartData = Object.entries(stressData.scenarios || {}).map(([scenario, data]) => ({
        scenario,
        unemployment: data.resulting_values?.unemployment || 0,
        oilPrice: data.resulting_values?.oil_price || 0,
        evImpact: data.resulting_values?.ev_sales_impact_pct || 0
    }))

    // Format clusters
    const clusterChartData = Object.entries(clusterData.commodity_clusters || {}).map(([name, data]) => ({
        name: name.replace('Cluster_', 'Group '),
        count: data.count,
        volatility: data.avg_volatility,
        members: data.members?.join(', ') || ''
    }))

    // Market anomaly data
    const marketAnomalies = anomalyData.market_wide_anomalies || []

    const sections = [
        { id: 'monte-carlo', label: '🎲 Monte Carlo', icon: '🎲' },
        { id: 'causality', label: '🔗 Causality', icon: '🔗' },
        { id: 'seasonality', label: '📅 Seasonality', icon: '📅' },
        { id: 'var', label: '📉 Value at Risk', icon: '📉' },
        { id: 'stress', label: '⚠️ Stress Tests', icon: '⚠️' },
        { id: 'clusters', label: '🎯 Clusters', icon: '🎯' },
        { id: 'anomalies', label: '🚨 Anomalies', icon: '🚨' },
        { id: 'regimes', label: '🔄 Regimes', icon: '🔄' }
    ]

    return (
        <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
            {/* Header */}
            <div style={{
                background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%)',
                borderRadius: '20px',
                padding: '2.5rem',
                marginBottom: '2rem',
                color: 'white',
                position: 'relative',
                overflow: 'hidden'
            }}>
                <div style={{ position: 'absolute', top: 0, right: 0, opacity: 0.1, fontSize: '15rem', lineHeight: 1 }}>🔬</div>
                <h1 style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>
                    Advanced Analytics
                </h1>
                <p style={{ fontSize: '1.1rem', opacity: 0.9, maxWidth: '600px' }}>
                    Deep ML insights from 10GB of data: Monte Carlo simulations, Granger causality,
                    Value at Risk, seasonality patterns, and stress testing scenarios.
                </p>
                <div style={{ display: 'flex', gap: '2rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', fontWeight: '700' }}>99.9%</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Prob. Battery &lt; $100 by 2030</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', fontWeight: '700' }}>6</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Proven Causal Relationships</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', fontWeight: '700' }}>14.1%</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Gas Price Seasonal Range</div>
                    </div>
                </div>
            </div>

            {/* Section Tabs */}
            <div style={{
                display: 'flex',
                gap: '0.5rem',
                marginBottom: '2rem',
                overflowX: 'auto',
                paddingBottom: '0.5rem'
            }}>
                {sections.map(section => (
                    <button
                        key={section.id}
                        onClick={() => setActiveSection(section.id)}
                        style={{
                            padding: '0.75rem 1.25rem',
                            borderRadius: '10px',
                            border: 'none',
                            background: activeSection === section.id
                                ? 'linear-gradient(135deg, #6366f1, #8b5cf6)'
                                : '#f1f5f9',
                            color: activeSection === section.id ? 'white' : '#475569',
                            fontWeight: '600',
                            cursor: 'pointer',
                            whiteSpace: 'nowrap',
                            transition: 'all 0.2s'
                        }}
                    >
                        {section.label}
                    </button>
                ))}
            </div>

            {/* Monte Carlo Section */}
            {activeSection === 'monte-carlo' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        🎲 Monte Carlo Simulations
                        <span style={{ fontSize: '0.9rem', fontWeight: '400', color: '#64748b' }}>
                            (1,000 simulated scenarios)
                        </span>
                    </h2>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem' }}>
                        {/* Battery Cost MC */}
                        <div style={{ background: 'white', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.08)' }}>
                            <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '1rem', color: '#1e293b' }}>
                                🔋 Battery Cost 2030 Probability Distribution
                            </h3>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                                <div style={{ background: '#f0fdf4', padding: '1rem', borderRadius: '10px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#16a34a' }}>${batteryMC['2030_median']}</div>
                                    <div style={{ fontSize: '0.8rem', color: '#15803d' }}>Median</div>
                                </div>
                                <div style={{ background: '#eff6ff', padding: '1rem', borderRadius: '10px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#2563eb' }}>${batteryMC['2030_p10']}-${batteryMC['2030_p90']}</div>
                                    <div style={{ fontSize: '0.8rem', color: '#1d4ed8' }}>P10-P90 Range</div>
                                </div>
                                <div style={{ background: '#fef3c7', padding: '1rem', borderRadius: '10px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#d97706' }}>{batteryMC.probability_below_100}%</div>
                                    <div style={{ fontSize: '0.8rem', color: '#b45309' }}>Prob &lt; $100</div>
                                </div>
                            </div>
                            <div style={{ background: 'linear-gradient(90deg, #10b981 0%, #10b981 99.9%, #ef4444 100%)', height: '20px', borderRadius: '10px', position: 'relative' }}>
                                <div style={{ position: 'absolute', left: '99.9%', top: '-5px', width: '3px', height: '30px', background: '#1e293b' }}></div>
                                <span style={{ position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', color: 'white', fontWeight: '600', fontSize: '0.75rem' }}>
                                    99.9% probability under $100/kWh
                                </span>
                            </div>
                        </div>

                        {/* EV Sales MC */}
                        <div style={{ background: 'white', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.08)' }}>
                            <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '1rem', color: '#1e293b' }}>
                                🚗 EV Sales 2030 Probability Distribution
                            </h3>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                                <div style={{ background: '#f0fdf4', padding: '1rem', borderRadius: '10px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#16a34a' }}>{evSalesMC['2030_median']}M</div>
                                    <div style={{ fontSize: '0.8rem', color: '#15803d' }}>Median</div>
                                </div>
                                <div style={{ background: '#eff6ff', padding: '1rem', borderRadius: '10px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#2563eb' }}>{evSalesMC['2030_p10']}-{evSalesMC['2030_p90']}M</div>
                                    <div style={{ fontSize: '0.8rem', color: '#1d4ed8' }}>P10-P90 Range</div>
                                </div>
                                <div style={{ background: '#fef3c7', padding: '1rem', borderRadius: '10px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#d97706' }}>{evSalesMC.probability_above_50m}%</div>
                                    <div style={{ fontSize: '0.8rem', color: '#b45309' }}>Prob &gt; 50M</div>
                                </div>
                            </div>
                            <p style={{ color: '#64748b', fontSize: '0.9rem' }}>
                                Current: {evSalesMC.current_sales_m}M units → Median 2030: {evSalesMC['2030_median']}M units
                                ({Math.round((evSalesMC['2030_median'] / evSalesMC.current_sales_m - 1) * 100)}% growth)
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Granger Causality Section */}
            {activeSection === 'causality' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '0.5rem' }}>
                        🔗 Granger Causality Analysis
                    </h2>
                    <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
                        Statistically proven leading indicators - these relationships have predictive power (p &lt; 0.05)
                    </p>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
                        {causalityData.filter(c => c.significant === 'True' || c.significant === true).map((item, idx) => (
                            <div key={idx} style={{
                                background: 'white',
                                borderRadius: '12px',
                                padding: '1.25rem',
                                boxShadow: '0 4px 15px rgba(0,0,0,0.08)',
                                borderLeft: `4px solid ${COLORS.primary}`
                            }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
                                    <span style={{ fontSize: '1.5rem' }}>📊</span>
                                    <div>
                                        <div style={{ fontWeight: '600', color: '#1e293b' }}>
                                            {item.cause.replace(/_/g, ' ')}
                                        </div>
                                        <div style={{ color: COLORS.primary, fontWeight: '600' }}>
                                            → {item.effect.replace(/_/g, ' ')}
                                        </div>
                                    </div>
                                </div>
                                <div style={{ display: 'flex', gap: '1rem' }}>
                                    <div style={{ background: '#f1f5f9', padding: '0.5rem 0.75rem', borderRadius: '8px' }}>
                                        <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Lag: </span>
                                        <span style={{ fontWeight: '600' }}>{item.best_lag_months} months</span>
                                    </div>
                                    <div style={{ background: '#dcfce7', padding: '0.5rem 0.75rem', borderRadius: '8px' }}>
                                        <span style={{ fontSize: '0.75rem', color: '#64748b' }}>p-value: </span>
                                        <span style={{ fontWeight: '600', color: '#16a34a' }}>{item.p_value.toFixed(4)}</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div style={{ background: '#f8fafc', borderRadius: '12px', padding: '1.5rem' }}>
                        <h3 style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '1rem' }}>What This Means</h3>
                        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'grid', gap: '0.75rem' }}>
                            <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
                                <span>📈</span>
                                <span><strong>Fed Funds Rate</strong> predicts housing starts 6 months ahead - watch rate decisions for real estate timing</span>
                            </li>
                            <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
                                <span>🚗</span>
                                <span><strong>Consumer Sentiment</strong> predicts vehicle sales 1 month ahead - sentiment surveys are actionable</span>
                            </li>
                            <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
                                <span>🏭</span>
                                <span><strong>Copper Price</strong> predicts industrial production 2 months ahead - "Dr. Copper" is real</span>
                            </li>
                        </ul>
                    </div>
                </div>
            )}

            {/* Seasonality Section */}
            {activeSection === 'seasonality' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '0.5rem' }}>
                        📅 Seasonality Patterns
                    </h2>
                    <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
                        Monthly patterns based on historical data - use for timing decisions
                    </p>

                    <div style={{ background: 'white', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.08)', marginBottom: '1.5rem' }}>
                        <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '1rem' }}>Gas Price Seasonality</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={seasonalityChartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                                <YAxis tickFormatter={v => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`} domain={[-10, 10]} />
                                <Tooltip formatter={(value) => [`${value > 0 ? '+' : ''}${value.toFixed(1)}%`, 'vs Average']} />
                                <ReferenceLine y={0} stroke="#94a3b8" />
                                <Bar dataKey="factor" radius={[4, 4, 0, 0]}>
                                    {seasonalityChartData.map((entry, index) => (
                                        <Cell key={index} fill={entry.factor >= 0 ? COLORS.danger : COLORS.success} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                        {Object.entries(seasonalityData).map(([name, data]) => (
                            <div key={name} style={{ background: 'white', borderRadius: '12px', padding: '1.25rem', boxShadow: '0 4px 15px rgba(0,0,0,0.08)' }}>
                                <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', color: '#1e293b' }}>
                                    {name.replace(/_/g, ' ')}
                                </h4>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                    <div>
                                        <span style={{ fontSize: '0.8rem', color: '#64748b' }}>Peak: </span>
                                        <span style={{ fontWeight: '600', color: COLORS.danger }}>{data.peak_month} ({data.peak_factor}%)</span>
                                    </div>
                                    <div>
                                        <span style={{ fontSize: '0.8rem', color: '#64748b' }}>Low: </span>
                                        <span style={{ fontWeight: '600', color: COLORS.success }}>{data.trough_month} ({data.trough_factor}%)</span>
                                    </div>
                                </div>
                                <div style={{ background: '#f1f5f9', padding: '0.5rem', borderRadius: '6px', textAlign: 'center' }}>
                                    <span style={{ fontSize: '0.85rem' }}>Range: <strong>{data.seasonal_range?.toFixed(1)}%</strong></span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Value at Risk Section */}
            {activeSection === 'var' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '0.5rem' }}>
                        📉 Value at Risk (VaR) Analysis
                    </h2>
                    <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
                        95% VaR = There's a 5% chance of losing more than this % in a single day
                    </p>

                    <div style={{ background: 'white', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.08)', marginBottom: '1.5rem' }}>
                        <ResponsiveContainer width="100%" height={400}>
                            <BarChart data={varChartData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis type="number" tickFormatter={v => `${v}%`} />
                                <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 11 }} />
                                <Tooltip
                                    formatter={(value, name) => [`${value.toFixed(2)}%`, name === 'var95' ? 'Daily VaR 95%' : 'Volatility']}
                                    labelFormatter={(label) => varChartData.find(d => d.name === label)?.fullName || label}
                                />
                                <Legend />
                                <Bar dataKey="var95" name="Daily VaR 95%" fill={COLORS.danger} radius={[0, 4, 4, 0]} />
                                <Bar dataKey="volatility" name="Annual Volatility" fill={COLORS.info} radius={[0, 4, 4, 0]} opacity={0.6} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                        <div style={{ background: 'linear-gradient(135deg, #fee2e2, #fef2f2)', borderRadius: '12px', padding: '1.25rem' }}>
                            <div style={{ fontSize: '0.85rem', color: '#b91c1c', marginBottom: '0.25rem' }}>Riskiest Commodity</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#dc2626' }}>Natural Gas</div>
                            <div style={{ fontSize: '0.9rem', color: '#991b1b' }}>-4.97% daily VaR</div>
                        </div>
                        <div style={{ background: 'linear-gradient(135deg, #dcfce7, #f0fdf4)', borderRadius: '12px', padding: '1.25rem' }}>
                            <div style={{ fontSize: '0.85rem', color: '#166534', marginBottom: '0.25rem' }}>Safest Commodity</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#16a34a' }}>Gold</div>
                            <div style={{ fontSize: '0.9rem', color: '#15803d' }}>-1.76% daily VaR</div>
                        </div>
                        <div style={{ background: 'linear-gradient(135deg, #e0e7ff, #eef2ff)', borderRadius: '12px', padding: '1.25rem' }}>
                            <div style={{ fontSize: '0.85rem', color: '#4338ca', marginBottom: '0.25rem' }}>Highest Volatility</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#4f46e5' }}>Crude Oil</div>
                            <div style={{ fontSize: '0.9rem', color: '#4338ca' }}>84.2% annualized</div>
                        </div>
                    </div>
                </div>
            )}

            {/* Stress Testing Section */}
            {activeSection === 'stress' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '0.5rem' }}>
                        ⚠️ Economic Stress Testing
                    </h2>
                    <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
                        How key variables would shift under different economic scenarios
                    </p>

                    <div style={{ background: 'white', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.08)', marginBottom: '1.5rem' }}>
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart data={stressChartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis dataKey="scenario" tick={{ fontSize: 12 }} />
                                <YAxis yAxisId="left" orientation="left" />
                                <YAxis yAxisId="right" orientation="right" />
                                <Tooltip />
                                <Legend />
                                <Bar yAxisId="left" dataKey="unemployment" name="Unemployment %" fill={COLORS.danger} radius={[4, 4, 0, 0]} />
                                <Bar yAxisId="right" dataKey="oilPrice" name="Oil Price $" fill={COLORS.secondary} radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
                        {Object.entries(stressData.scenarios || {}).map(([scenario, data]) => (
                            <div key={scenario} style={{
                                background: 'white',
                                borderRadius: '12px',
                                padding: '1.25rem',
                                boxShadow: '0 4px 15px rgba(0,0,0,0.08)',
                                borderTop: `4px solid ${scenario === 'Recession' ? COLORS.danger :
                                        scenario === 'Financial Crisis' ? '#991b1b' :
                                            scenario === 'Oil Shock' ? COLORS.warning :
                                                COLORS.success
                                    }`
                            }}>
                                <h4 style={{ fontWeight: '700', marginBottom: '0.75rem', fontSize: '1.1rem' }}>{scenario}</h4>
                                <div style={{ display: 'grid', gap: '0.5rem', fontSize: '0.9rem' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span style={{ color: '#64748b' }}>Unemployment:</span>
                                        <span style={{ fontWeight: '600' }}>{data.resulting_values?.unemployment?.toFixed(1)}%</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span style={{ color: '#64748b' }}>Oil Price:</span>
                                        <span style={{ fontWeight: '600' }}>${data.resulting_values?.oil_price?.toFixed(0)}</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span style={{ color: '#64748b' }}>EV Sales Impact:</span>
                                        <span style={{ fontWeight: '600', color: data.resulting_values?.ev_sales_impact_pct > 0 ? COLORS.success : COLORS.danger }}>
                                            {data.resulting_values?.ev_sales_impact_pct > 0 ? '+' : ''}{data.resulting_values?.ev_sales_impact_pct}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Clusters Section */}
            {activeSection === 'clusters' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '0.5rem' }}>
                        🎯 Commodity Clustering
                    </h2>
                    <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
                        Commodities grouped by similar volatility and return patterns (K-Means clustering)
                    </p>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
                        {clusterChartData.map((cluster, idx) => (
                            <div key={cluster.name} style={{
                                background: 'white',
                                borderRadius: '12px',
                                padding: '1.5rem',
                                boxShadow: '0 4px 15px rgba(0,0,0,0.08)',
                                borderLeft: `5px solid ${[COLORS.success, COLORS.danger, COLORS.warning, COLORS.info][idx]}`
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                    <h4 style={{ fontWeight: '700', fontSize: '1.1rem' }}>{cluster.name}</h4>
                                    <span style={{ background: '#f1f5f9', padding: '0.25rem 0.75rem', borderRadius: '20px', fontSize: '0.85rem' }}>
                                        {cluster.count} commodities
                                    </span>
                                </div>
                                <div style={{ marginBottom: '1rem' }}>
                                    <span style={{ fontSize: '0.85rem', color: '#64748b' }}>Avg Volatility: </span>
                                    <span style={{ fontSize: '1.25rem', fontWeight: '700', color: cluster.volatility > 50 ? COLORS.danger : COLORS.primary }}>
                                        {cluster.volatility?.toFixed(1)}%
                                    </span>
                                </div>
                                <div style={{ fontSize: '0.85rem', color: '#64748b', lineHeight: '1.5' }}>
                                    {cluster.members}
                                </div>
                            </div>
                        ))}
                    </div>

                    {clusterData.pca_variance_explained && (
                        <div style={{ background: '#f8fafc', borderRadius: '12px', padding: '1.25rem', marginTop: '1.5rem' }}>
                            <h4 style={{ fontWeight: '600', marginBottom: '0.5rem' }}>PCA Analysis</h4>
                            <p style={{ color: '#64748b', fontSize: '0.9rem' }}>
                                The first two principal components explain{' '}
                                <strong>{clusterData.pca_variance_explained.PC1 + clusterData.pca_variance_explained.PC2}%</strong> of the variance
                                (PC1: {clusterData.pca_variance_explained.PC1}%, PC2: {clusterData.pca_variance_explained.PC2}%)
                            </p>
                        </div>
                    )}
                </div>
            )}

            {/* Anomalies Section */}
            {activeSection === 'anomalies' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '0.5rem' }}>
                        🚨 Market-Wide Anomaly Detection
                    </h2>
                    <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
                        Days when multiple commodities moved abnormally together (using Isolation Forest)
                    </p>

                    <div style={{ background: 'white', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.08)', marginBottom: '1.5rem' }}>
                        <h3 style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '1rem' }}>Top Market-Wide Anomaly Days</h3>
                        <div style={{ display: 'grid', gap: '0.75rem' }}>
                            {marketAnomalies.slice(0, 10).map((day, idx) => (
                                <div key={idx} style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    padding: '0.75rem 1rem',
                                    background: idx < 3 ? '#fef2f2' : '#f8fafc',
                                    borderRadius: '8px',
                                    borderLeft: `4px solid ${idx < 3 ? COLORS.danger : COLORS.warning}`
                                }}>
                                    <div>
                                        <span style={{ fontWeight: '600' }}>{day.date}</span>
                                        <span style={{ marginLeft: '0.5rem', fontSize: '0.85rem', color: '#64748b' }}>
                                            {day.date.includes('2008') ? '(Financial Crisis)' :
                                                day.date.includes('2020') ? '(COVID-19)' : ''}
                                        </span>
                                    </div>
                                    <div style={{
                                        background: idx < 3 ? COLORS.danger : COLORS.warning,
                                        color: 'white',
                                        padding: '0.25rem 0.75rem',
                                        borderRadius: '20px',
                                        fontSize: '0.85rem',
                                        fontWeight: '600'
                                    }}>
                                        {day.assets_affected} assets affected
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div style={{ background: '#fef3c7', borderRadius: '12px', padding: '1.25rem' }}>
                        <h4 style={{ fontWeight: '600', marginBottom: '0.5rem', color: '#92400e' }}>💡 Key Insight</h4>
                        <p style={{ color: '#78350f', fontSize: '0.95rem' }}>
                            The 2008 Financial Crisis produced the most correlated market crashes in history,
                            with 17 commodities moving abnormally on single days. COVID-19 also triggered
                            correlated anomalies but with slightly less severity.
                        </p>
                    </div>
                </div>
            )}

            {/* Regimes Section */}
            {activeSection === 'regimes' && (
                <div>
                    <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '0.5rem' }}>
                        🔄 Market Regime Detection
                    </h2>
                    <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
                        Current volatility regime for key assets (using Gaussian Mixture Models)
                    </p>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
                        {Object.entries(regimeData).map(([asset, data]) => (
                            <div key={asset} style={{
                                background: 'white',
                                borderRadius: '12px',
                                padding: '1.5rem',
                                boxShadow: '0 4px 15px rgba(0,0,0,0.08)'
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                    <h4 style={{ fontWeight: '700', fontSize: '1.1rem' }}>{asset.replace(/_/g, ' ')}</h4>
                                    <span style={{
                                        background: data.current_regime === 'Low Vol' ? '#dcfce7' :
                                            data.current_regime === 'High Vol' ? '#fee2e2' : '#fef3c7',
                                        color: data.current_regime === 'Low Vol' ? '#166534' :
                                            data.current_regime === 'High Vol' ? '#dc2626' : '#92400e',
                                        padding: '0.25rem 0.75rem',
                                        borderRadius: '20px',
                                        fontSize: '0.85rem',
                                        fontWeight: '600'
                                    }}>
                                        {data.current_regime}
                                    </span>
                                </div>
                                <div style={{ display: 'grid', gap: '0.5rem' }}>
                                    {data.regimes?.map((regime, idx) => (
                                        <div key={idx} style={{
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            padding: '0.5rem',
                                            background: regime.label === data.current_regime ? '#f0fdf4' : '#f8fafc',
                                            borderRadius: '6px',
                                            fontSize: '0.9rem'
                                        }}>
                                            <span style={{ fontWeight: regime.label === data.current_regime ? '600' : '400' }}>
                                                {regime.label}
                                            </span>
                                            <span style={{ color: '#64748b' }}>
                                                {regime.mean_vol?.toFixed(1)}% vol ({regime.frequency_pct?.toFixed(0)}% of time)
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>

                    <div style={{ background: '#f0fdf4', borderRadius: '12px', padding: '1.25rem', marginTop: '1.5rem' }}>
                        <h4 style={{ fontWeight: '600', marginBottom: '0.5rem', color: '#166534' }}>✅ Current Market State</h4>
                        <p style={{ color: '#15803d', fontSize: '0.95rem' }}>
                            All major assets are currently in <strong>Low Volatility</strong> regimes.
                            This suggests stable markets, but historically low-vol regimes eventually transition
                            to higher volatility. Be prepared for regime changes.
                        </p>
                    </div>
                </div>
            )}
        </div>
    )
}
