import { useState } from 'react';
import {
    AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart,
    ReferenceLine, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import ChartModal from '../components/ChartModal';

// Import the deep analysis results
import analysisData from '../data/deep_analysis_results.json';

export default function DeepAnalysis() {
    const [modalData, setModalData] = useState(null);

    const openModal = (title, content) => {
        setModalData({ title, content });
    };

    // Prepare job category data for chart
    const jobCategoryData = Object.entries(analysisData.job_market.job_categories).map(([name, data]) => ({
        name: name.length > 20 ? name.substring(0, 18) + '...' : name,
        fullName: name,
        demandChange: data.demand_change,
        fill: data.demand_change > 0 ? '#10b981' : data.demand_change > -30 ? '#f59e0b' : '#ef4444'
    })).sort((a, b) => b.demandChange - a.demandChange);

    // Geographic risk data
    const geoRiskData = Object.entries(analysisData.geographic_risk.risk_scores).map(([name, score]) => ({
        resource: name,
        score: score,
        fill: score > 70 ? '#ef4444' : score > 40 ? '#f59e0b' : '#10b981'
    })).sort((a, b) => b.score - a.score);

    // Copper trajectory data (sample every 6 months for readability)
    const copperTrajectory = analysisData.trajectory_data.copper_deficit.dates
        .filter((_, i) => i % 6 === 0 || i === analysisData.trajectory_data.copper_deficit.dates.length - 1)
        .map((date, i) => ({
            date: date.substring(0, 7),
            deficit: analysisData.trajectory_data.copper_deficit.values.filter((_, j) => j % 6 === 0 || j === analysisData.trajectory_data.copper_deficit.values.length - 1)[i]
        }));

    // Cross-domain correlation data
    const correlationData = analysisData.cross_domain_correlations.map(c => ({
        ...c,
        absCorr: Math.abs(c.correlation),
        color: c.correlation > 0 ? '#10b981' : '#ef4444'
    }));

    // Feature importance from attention model
    const featureImportance = Object.entries(analysisData.deep_learning.attention.feature_importance)
        .map(([feature, importance]) => ({
            feature: feature.replace(/_/g, ' ').replace('mmt', '').replace('gw', ''),
            importance: importance * 100,
            fill: importance > 0.5 ? '#8b5cf6' : importance > 0.1 ? '#3b82f6' : '#6b7280'
        })).sort((a, b) => b.importance - a.importance);

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🧠 Deep ML Cross-Domain Analysis</h1>
                <p className="page-subtitle">
                    GPU-accelerated analysis (RTX 3060) • {analysisData.dataset_info.time_points} time points • {analysisData.dataset_info.variables} variables
                </p>
            </header>

            {/* Key Stats Banner */}
            <div className="stats-grid" style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '16px',
                marginBottom: '24px'
            }}>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #ef44441a, #dc26261a)', borderLeft: '4px solid #ef4444' }}>
                    <div className="stat-label">Copper Deficit 2035</div>
                    <div className="stat-value" style={{ color: '#ef4444' }}>{analysisData.copper_collision.combined_peak_2035} MMT</div>
                    <div className="stat-detail">EV + AI combined demand</div>
                </div>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #10b9811a, #0596591a)', borderLeft: '4px solid #10b981' }}>
                    <div className="stat-label">Net Carbon 2025</div>
                    <div className="stat-value" style={{ color: '#10b981' }}>{analysisData.energy_net_effect.net_carbon_2025_mt} MT</div>
                    <div className="stat-detail">AI is carbon NEGATIVE (saves more)</div>
                </div>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #f59e0b1a, #d978021a)', borderLeft: '4px solid #f59e0b' }}>
                    <div className="stat-label">Jobs at Risk</div>
                    <div className="stat-value" style={{ color: '#f59e0b' }}>{analysisData.job_market.at_risk_jobs_2025_m}M</div>
                    <div className="stat-detail">vs {analysisData.job_market.ai_jobs_2025_k}K AI jobs created</div>
                </div>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #8b5cf61a, #7c3aed1a)', borderLeft: '4px solid #8b5cf6' }}>
                    <div className="stat-label">Price Lag</div>
                    <div className="stat-value" style={{ color: '#8b5cf6' }}>{analysisData.price_transmission.optimal_lag_months} months</div>
                    <div className="stat-detail">Component to Consumer delay</div>
                </div>
            </div>

            {/* Question 1: Copper Collision */}
            <div className="chart-card" onClick={() => openModal(
                'EV + AI Copper Collision Analysis',
                `Analysis of combined copper demand from electric vehicles and AI data centers. The correlation between EV and AI copper demand is nearly perfect (r=${analysisData.copper_collision.ev_ai_correlation}), indicating they are on converging growth trajectories. By 2035, combined demand reaches ${analysisData.copper_collision.combined_peak_2035} MMT while supply struggles to keep pace. Our GradientBoosting model achieves R2=${analysisData.copper_collision.price_prediction_r2} for price prediction, forecasting copper at $${analysisData.copper_collision.price_forecast_2030}/ton by 2030.`
            )}>
                <h3 className="chart-title">🔴 Question 1: EV + AI Copper Collision</h3>
                <p style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '16px' }}>
                    When do EV and AI demand exceed copper supply?
                </p>

                <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={copperTrajectory}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                        <XAxis dataKey="date" stroke="#71717a" tick={{ fontSize: 10 }} />
                        <YAxis stroke="#71717a" domain={[0, 30]} label={{ value: 'Deficit (MMT)', angle: -90, position: 'insideLeft', fill: '#71717a' }} />
                        <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                        <ReferenceLine y={5} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'Critical Deficit', fill: '#ef4444', fontSize: 10 }} />
                        <Area type="monotone" dataKey="deficit" stroke="#f59e0b" fill="url(#copperGradient)" strokeWidth={2} />
                        <defs>
                            <linearGradient id="copperGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8} />
                                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.1} />
                            </linearGradient>
                        </defs>
                    </AreaChart>
                </ResponsiveContainer>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginTop: '16px' }}>
                    <div style={{ textAlign: 'center', padding: '12px', background: 'rgba(245, 158, 11, 0.1)', borderRadius: '8px' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#f59e0b' }}>r = {analysisData.copper_collision.ev_ai_correlation}</div>
                        <div style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.6)' }}>EV-AI Correlation</div>
                    </div>
                    <div style={{ textAlign: 'center', padding: '12px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#ef4444' }}>${(analysisData.copper_collision.price_forecast_2030 / 1000).toFixed(1)}K</div>
                        <div style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.6)' }}>Price/ton by 2030</div>
                    </div>
                    <div style={{ textAlign: 'center', padding: '12px', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '8px' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#8b5cf6' }}>{analysisData.copper_collision.price_prediction_r2}</div>
                        <div style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.6)' }}>Model R2 Score</div>
                    </div>
                </div>
            </div>

            {/* Question 2: Energy Net Effect */}
            <div className="chart-card" style={{
                background: analysisData.energy_net_effect.net_carbon_2025_mt < 0
                    ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%)'
                    : undefined,
                border: analysisData.energy_net_effect.net_carbon_2025_mt < 0 ? '2px solid rgba(16, 185, 129, 0.3)' : undefined
            }} onClick={() => openModal(
                'Energy Net Effect Analysis',
                `Despite consuming ${analysisData.energy_net_effect.ai_carbon_2025_mt} MT CO2 in 2025, AI systems save ${analysisData.energy_net_effect.ai_savings_2025_mt} MT through efficiency improvements (logistics, grid, manufacturing). The net result is ${analysisData.energy_net_effect.net_carbon_2025_mt} MT - AI is a CARBON SAVER when considering induced efficiencies. Efficiency gains grow from ${analysisData.energy_net_effect.efficiency_growth_2025_2035}.`
            )}>
                <h3 className="chart-title">⚡ Question 2: AI Energy - Net Carbon Effect</h3>
                <p style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '16px' }}>
                    Does AI consume more carbon than it saves?
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '20px', marginBottom: '20px' }}>
                    <div style={{ padding: '20px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '12px', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                        <h4 style={{ color: '#ef4444', marginBottom: '12px' }}>🔥 AI Consumption</h4>
                        <div style={{ fontSize: '2rem', fontWeight: '700', color: '#ef4444' }}>
                            {analysisData.energy_net_effect.ai_carbon_2025_mt} MT
                        </div>
                        <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.85rem' }}>CO2 from AI infrastructure</div>
                        <div style={{ marginTop: '12px', color: '#ef4444' }}>
                            Power: {analysisData.energy_net_effect.ai_power_2025_gw} GW to {analysisData.energy_net_effect.ai_power_2035_gw} GW
                        </div>
                    </div>
                    <div style={{ padding: '20px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '12px', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                        <h4 style={{ color: '#10b981', marginBottom: '12px' }}>💚 AI Savings</h4>
                        <div style={{ fontSize: '2rem', fontWeight: '700', color: '#10b981' }}>
                            {analysisData.energy_net_effect.ai_savings_2025_mt} MT
                        </div>
                        <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.85rem' }}>CO2 saved via efficiency</div>
                        <div style={{ marginTop: '12px', color: '#10b981' }}>
                            Efficiency: {analysisData.energy_net_effect.efficiency_growth_2025_2035}
                        </div>
                    </div>
                </div>

                <div style={{
                    padding: '24px',
                    background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1))',
                    borderRadius: '12px',
                    textAlign: 'center',
                    border: '2px solid rgba(16, 185, 129, 0.4)'
                }}>
                    <div style={{ fontSize: '0.9rem', color: 'rgba(255,255,255,0.7)', marginBottom: '8px' }}>NET CARBON IMPACT</div>
                    <div style={{ fontSize: '3rem', fontWeight: '800', color: '#10b981' }}>
                        {analysisData.energy_net_effect.net_carbon_2025_mt} MT
                    </div>
                    <div style={{ fontSize: '1.2rem', color: '#10b981', fontWeight: '600' }}>
                        AI IS A NET CARBON SAVER
                    </div>
                </div>
            </div>

            {/* Question 3: Job Market */}
            <div className="chart-card" onClick={() => openModal(
                'Job Market Impact Analysis',
                `Based on research from multiple sources, AI-skilled roles are growing 80%+ while entry-level and generalist roles are declining 25-65%. In 2025, approximately ${analysisData.job_market.ai_jobs_2025_k}K new AI jobs are created, but ${analysisData.job_market.at_risk_jobs_2025_m}M traditional tech jobs are at risk. This represents a significant restructuring of the tech labor market, not just job loss.`
            )}>
                <h3 className="chart-title">💼 Question 3: Job Market Impact by Role</h3>
                <p style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '16px' }}>
                    Which jobs grow vs decline with AI?
                </p>

                <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={jobCategoryData} layout="vertical" margin={{ left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                        <XAxis type="number" stroke="#71717a" domain={[-80, 100]} tickFormatter={(v) => `${v}%`} />
                        <YAxis dataKey="name" type="category" stroke="#71717a" width={130} tick={{ fontSize: 11 }} />
                        <Tooltip
                            contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                            formatter={(value) => [`${value}%`, 'Demand Change']}
                            labelFormatter={(label) => jobCategoryData.find(j => j.name === label)?.fullName || label}
                        />
                        <ReferenceLine x={0} stroke="#71717a" />
                        <Bar dataKey="demandChange" radius={[0, 4, 4, 0]}>
                            {jobCategoryData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>

                <div className="chart-insight">
                    <strong>Key Finding:</strong> {analysisData.job_market.key_insight}
                </div>
            </div>

            {/* Question 4: Geographic Risk */}
            <div className="chart-card" onClick={() => openModal(
                'Geographic Supply Chain Risk Analysis',
                `Supply chain concentration is a major risk. Taiwan controls 92% of advanced semiconductors - a disruption would cause $10T in global GDP losses. South Korea controls 95% of HBM memory (critical for AI). Diversification efforts wont meaningfully reduce risk until 2030-2035.`
            )}>
                <h3 className="chart-title">🌍 Question 4: Geographic Risk Mapping</h3>
                <p style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '16px' }}>
                    Where are the supply chain single points of failure?
                </p>

                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={geoRiskData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                        <XAxis type="number" stroke="#71717a" domain={[0, 100]} />
                        <YAxis dataKey="resource" type="category" stroke="#71717a" width={160} tick={{ fontSize: 11 }} />
                        <Tooltip
                            contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                            formatter={(value) => [value.toFixed(1), 'Risk Score']}
                        />
                        <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                            {geoRiskData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginTop: '16px' }}>
                    {Object.entries(analysisData.geographic_risk.geographic_risks).slice(0, 3).map(([name, data]) => (
                        <div key={name} style={{
                            padding: '12px',
                            background: 'rgba(239, 68, 68, 0.1)',
                            borderRadius: '8px',
                            borderLeft: '3px solid #ef4444'
                        }}>
                            <div style={{ fontWeight: '600', color: '#ef4444', marginBottom: '4px' }}>{name}</div>
                            <div style={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.7)' }}>
                                {data.primary_location} ({data.concentration}%)
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.5)', marginTop: '4px' }}>
                                {data.economic_impact_if_disrupted}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Question 5: Price Transmission */}
            <div className="chart-card" onClick={() => openModal(
                'Price Transmission Analysis',
                `Component price changes take ${analysisData.price_transmission.optimal_lag_months} months to fully transmit to consumer products. Memory prices increased ${analysisData.price_transmission.memory_increase_2020_2025} from 2020-2025, while laptops increased ${analysisData.price_transmission.laptop_increase_2020_2025}. The price elasticity is ${analysisData.price_transmission.price_elasticity} - meaning a 10% component increase leads to about 3% consumer price increase.`
            )}>
                <h3 className="chart-title">💰 Question 5: Price Transmission Lag</h3>
                <p style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '16px' }}>
                    How long until component prices hit consumers?
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '20px' }}>
                    <div style={{ textAlign: 'center', padding: '20px', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '12px' }}>
                        <div style={{ fontSize: '3rem', fontWeight: '800', color: '#8b5cf6' }}>
                            {analysisData.price_transmission.optimal_lag_months}
                        </div>
                        <div style={{ color: 'rgba(255,255,255,0.7)' }}>Months Lag</div>
                        <div style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.5)', marginTop: '4px' }}>
                            r = {analysisData.price_transmission.lag_correlation}
                        </div>
                    </div>
                    <div style={{ textAlign: 'center', padding: '20px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', fontWeight: '700', color: '#ef4444' }}>
                            {analysisData.price_transmission.memory_increase_2020_2025}
                        </div>
                        <div style={{ color: 'rgba(255,255,255,0.7)' }}>Memory Increase</div>
                        <div style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.5)', marginTop: '4px' }}>
                            2020 to 2025
                        </div>
                    </div>
                    <div style={{ textAlign: 'center', padding: '20px', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', fontWeight: '700', color: '#3b82f6' }}>
                            {analysisData.price_transmission.laptop_increase_2020_2025}
                        </div>
                        <div style={{ color: 'rgba(255,255,255,0.7)' }}>Laptop Increase</div>
                        <div style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.5)', marginTop: '4px' }}>
                            Elasticity: {analysisData.price_transmission.price_elasticity}
                        </div>
                    </div>
                </div>

                <div className="chart-insight">
                    <strong>Implication:</strong> Memory prices up 775% means Expect laptops +47% with ~6 month delay.
                    If DDR5 at $27/GB today, consumer PC prices reflect this impact by mid-2026.
                </div>
            </div>

            {/* Deep Learning Results */}
            <div className="chart-card" style={{
                background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%)',
                border: '2px solid rgba(139, 92, 246, 0.3)'
            }}>
                <h3 className="chart-title">🧠 Deep Learning Model Results</h3>
                <p style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '20px' }}>
                    GPU-accelerated analysis using PyTorch + RTX 3060
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
                    {/* LSTM Results */}
                    <div style={{ padding: '16px', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '12px' }}>
                        <h4 style={{ color: '#8b5cf6', marginBottom: '12px' }}>🔄 LSTM Time Series</h4>
                        <div style={{ fontSize: '0.9rem', color: 'rgba(255,255,255,0.8)' }}>
                            <div>Epochs: {analysisData.deep_learning.lstm.epochs}</div>
                            <div>MAE: ${analysisData.deep_learning.lstm.mae.toLocaleString()}</div>
                            <div style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.5)', marginTop: '8px' }}>
                                Features: {analysisData.deep_learning.lstm.features.join(', ')}
                            </div>
                        </div>
                    </div>

                    {/* VAE Anomaly */}
                    <div style={{ padding: '16px', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '12px' }}>
                        <h4 style={{ color: '#3b82f6', marginBottom: '12px' }}>🔍 VAE Anomaly Detection</h4>
                        <div style={{ fontSize: '0.9rem', color: 'rgba(255,255,255,0.8)' }}>
                            <div>Anomalies Found: {analysisData.deep_learning.vae_anomaly.anomaly_count}</div>
                            <div>Threshold: {analysisData.deep_learning.vae_anomaly.threshold}</div>
                            <div style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.5)', marginTop: '8px' }}>
                                Anomaly Years: {analysisData.deep_learning.vae_anomaly.anomaly_dates.slice(0, 3).map(d => d.substring(0, 4)).join(', ')}...
                            </div>
                        </div>
                    </div>

                    {/* Attention Feature Importance */}
                    <div style={{ padding: '16px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '12px', gridColumn: 'span 2' }}>
                        <h4 style={{ color: '#10b981', marginBottom: '12px' }}>🎯 Attention-Based Feature Importance</h4>
                        <ResponsiveContainer width="100%" height={150}>
                            <BarChart data={featureImportance} layout="vertical">
                                <XAxis type="number" stroke="#71717a" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                                <YAxis dataKey="feature" type="category" stroke="#71717a" width={100} tick={{ fontSize: 10 }} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(value) => [`${value.toFixed(1)}%`, 'Importance']}
                                />
                                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                                    {featureImportance.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.fill} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                        <div className="chart-insight" style={{ marginTop: '8px' }}>
                            <strong>Key Finding:</strong> AI Power is 94%+ responsible for copper price predictions - confirming AI as the dominant demand driver.
                        </div>
                    </div>
                </div>
            </div>

            {/* Key Insights Summary */}
            <div className="chart-card" style={{
                background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(234, 88, 12, 0.1) 100%)',
                border: '2px solid rgba(245, 158, 11, 0.3)'
            }}>
                <h3 className="chart-title">📋 Key Insights from Deep Analysis</h3>
                <div style={{ padding: '16px' }}>
                    <ul style={{
                        color: 'rgba(255,255,255,0.9)',
                        lineHeight: '2',
                        fontSize: '1rem',
                        paddingLeft: '20px'
                    }}>
                        {analysisData.key_insights.map((insight, idx) => (
                            <li key={idx} style={{ marginBottom: '8px' }}>{insight}</li>
                        ))}
                    </ul>
                </div>
            </div>

            {/* Cross-Domain Correlations */}
            <div className="chart-card">
                <h3 className="chart-title">🔗 Cross-Domain Correlations</h3>
                <p style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '16px' }}>
                    Surprising relationships discovered across different domains
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px' }}>
                    {correlationData.map((c, idx) => (
                        <div key={idx} style={{
                            padding: '16px',
                            background: c.correlation > 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                            borderRadius: '8px',
                            borderLeft: `4px solid ${c.color}`
                        }}>
                            <div style={{ fontWeight: '600', color: c.color, marginBottom: '4px' }}>
                                r = {c.correlation.toFixed(3)}
                            </div>
                            <div style={{ fontSize: '0.9rem', color: 'rgba(255,255,255,0.8)' }}>
                                {c.variables}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.5)', marginTop: '4px' }}>
                                {c.strength} {c.correlation > 0 ? 'Positive' : 'Negative'}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {modalData && (
                <div className="modal-backdrop" onClick={() => setModalData(null)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <h3>{modalData.title}</h3>
                        <p style={{ lineHeight: '1.8' }}>{modalData.content}</p>
                        <button onClick={() => setModalData(null)}>Close</button>
                    </div>
                </div>
            )}
        </div>
    );
}
