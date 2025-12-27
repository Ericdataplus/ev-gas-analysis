import { useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    LineChart, Line, AreaChart, Area, Cell
} from 'recharts';

// Import both analysis files
import signalsData from '../data/ai_timeline_signals.json';
import deepDiveData from '../data/ai_timeline_deep_dive.json';
import metalsData from '../data/ai_metals_analysis.json';

const COLORS = {
    green: '#10b981',
    yellow: '#f59e0b',
    red: '#ef4444',
    blue: '#3b82f6',
    purple: '#8b5cf6',
    cyan: '#06b6d4'
};

export default function AITimeline() {
    const [activeTab, setActiveTab] = useState('overview');

    // Data from deep dive
    const dd1 = deepDiveData.deep_analysis?.dd1_ai_boom_timing || {};
    const dd2 = deepDiveData.deep_analysis?.dd2_lead_lag || {};
    const dd3 = deepDiveData.deep_analysis?.dd3_constraint_acceleration || {};
    const dd4 = deepDiveData.deep_analysis?.dd4_breaking_points || {};
    const dd5 = deepDiveData.deep_analysis?.dd5_infrastructure_rate || {};
    const dd6 = deepDiveData.deep_analysis?.dd6_forecasting || {};
    const keyFindings = deepDiveData.key_findings || [];

    // Prepare chart data
    const copperComparisonData = [
        { name: 'Pre-ChatGPT', value: dd1.pre_chatgpt_avg || 3698, fill: COLORS.blue },
        { name: 'Post-ChatGPT', value: dd1.post_chatgpt_avg || 9021, fill: COLORS.purple }
    ];

    const leadLagData = Object.entries(dd2).map(([key, val]) => ({
        name: key.replace('_vs_', ' → ').replace(/_/g, ' '),
        correlation: Math.abs(val.best_correlation || 0) * 100,
        lag: val.optimal_lag || 0,
        interpretation: val.interpretation
    }));

    const constraintData = Object.entries(dd3).map(([key, val]) => ({
        name: key.replace('_', ' ').toUpperCase(),
        growth: val.recent_avg_growth_pct || 0,
        trend: val.trend
    }));

    return (
        <div className="dashboard">
            {/* Header */}
            <div className="page-header" style={{ marginBottom: '2rem' }}>
                <h1 style={{ fontSize: '2rem', fontWeight: '700', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span style={{ fontSize: '2.5rem' }}>🤖</span> AI Timeline Signals
                </h1>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', maxWidth: '800px' }}>
                    Using <strong>real economic data</strong> to understand physical constraints on AI development.
                    Can infrastructure scale fast enough for exponential AI growth?
                </p>
            </div>

            {/* Key Metrics Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
                <div style={{ background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)', borderRadius: '16px', padding: '1.25rem', color: 'white' }}>
                    <div style={{ fontSize: '2.25rem', fontWeight: '700' }}>+144%</div>
                    <div style={{ opacity: 0.9, fontSize: '0.85rem' }}>Copper Since ChatGPT</div>
                </div>
                <div style={{ background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)', borderRadius: '16px', padding: '1.25rem', color: 'white' }}>
                    <div style={{ fontSize: '2.25rem', fontWeight: '700' }}>-3.9%</div>
                    <div style={{ opacity: 0.9, fontSize: '0.85rem' }}>From All-Time High</div>
                </div>
                <div style={{ background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)', borderRadius: '16px', padding: '1.25rem', color: 'white' }}>
                    <div style={{ fontSize: '2.25rem', fontWeight: '700' }}>12mo</div>
                    <div style={{ opacity: 0.9, fontSize: '0.85rem' }}>Fed Rate Lead Time</div>
                </div>
                <div style={{ background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', borderRadius: '16px', padding: '1.25rem', color: 'white' }}>
                    <div style={{ fontSize: '2.25rem', fontWeight: '700' }}>+85%</div>
                    <div style={{ opacity: 0.9, fontSize: '0.85rem' }}>Infrastructure Intensity</div>
                </div>
            </div>

            {/* Key Findings */}
            <div className="card" style={{ background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(99, 102, 241, 0.1) 100%)', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem', borderLeft: '4px solid #8b5cf6' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    🎯 Key AI Timeline Findings
                </h2>
                <div style={{ display: 'grid', gap: '0.75rem' }}>
                    {keyFindings.map((finding, i) => (
                        <div key={i} style={{
                            background: 'rgba(255,255,255,0.05)',
                            padding: '0.875rem 1rem',
                            borderRadius: '10px',
                            fontSize: '0.95rem',
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: '0.75rem'
                        }}>
                            <span style={{ fontSize: '1.1rem' }}>{i === 0 ? '📊' : i === 1 ? '📈' : i === 2 ? '🔴' : i === 3 ? '🟡' : '✅'}</span>
                            <span>{finding.replace(/^[📊📈🔴🟡✅⚠️]\s*/, '')}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* ChatGPT Era Impact */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
                <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem' }}>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '0.5rem' }}>🤖 ChatGPT Era Impact on Copper</h2>
                    <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.85rem' }}>
                        Pre-ChatGPT (2015-Nov 2022) vs Post-ChatGPT (Nov 2022+)
                    </p>
                    <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={copperComparisonData} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis type="number" tickFormatter={v => `$${(v / 1000).toFixed(1)}k`} />
                            <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={100} />
                            <Tooltip
                                formatter={(value) => [`$${value.toLocaleString()}/ton`, 'Avg Price']}
                                contentStyle={{ background: '#1e1e2e', border: 'none', borderRadius: '8px' }}
                            />
                            <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                                {copperComparisonData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.fill} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                    <div style={{
                        marginTop: '1rem',
                        padding: '1rem',
                        background: 'rgba(139, 92, 246, 0.15)',
                        borderRadius: '10px',
                        textAlign: 'center'
                    }}>
                        <span style={{ fontSize: '1.5rem', fontWeight: '700', color: COLORS.purple }}>+{dd1.chatgpt_era_change_pct}%</span>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Price increase since AI boom</div>
                    </div>
                </div>

                {/* Current Status */}
                <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem' }}>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '0.5rem' }}>📊 Current Copper Status</h2>
                    <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.85rem' }}>
                        Distance from historical breaking point
                    </p>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                            <span>All-Time High (Mar 2022)</span>
                            <span style={{ fontWeight: '600' }}>${dd4.all_time_high?.toLocaleString()}/ton</span>
                        </div>
                        <div style={{ height: '12px', background: 'rgba(255,255,255,0.1)', borderRadius: '6px', overflow: 'hidden' }}>
                            <div style={{
                                height: '100%',
                                width: `${(dd4.current_price / dd4.all_time_high * 100)}%`,
                                background: 'linear-gradient(90deg, #10b981, #f59e0b, #ef4444)',
                                borderRadius: '6px',
                                transition: 'width 0.5s ease'
                            }} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.85rem' }}>
                            <span>Current: ${dd4.current_price?.toLocaleString()}/ton</span>
                            <span style={{ color: COLORS.yellow }}>{dd4.pct_from_ath}% from ATH</span>
                        </div>
                    </div>

                    <div style={{
                        padding: '1rem',
                        background: 'rgba(239, 68, 68, 0.15)',
                        borderRadius: '10px',
                        borderLeft: '4px solid #ef4444'
                    }}>
                        <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>⚠️ Near Stress Levels</div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                            Copper is within {Math.abs(dd4.pct_from_ath)}% of all-time high.
                            Historical peaks often trigger demand destruction.
                        </div>
                    </div>
                </div>
            </div>

            {/* Lead/Lag Analysis */}
            <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '0.5rem' }}>📈 Lead/Lag Analysis: What Predicts What?</h2>
                <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem', fontSize: '0.85rem' }}>
                    Understanding causal relationships between economic indicators
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
                    {Object.entries(dd2).map(([key, val]) => {
                        const parts = key.split('_vs_');
                        const isLeading = val.optimal_lag !== 0;
                        const color = isLeading ? (val.optimal_lag < 0 ? COLORS.blue : COLORS.purple) : COLORS.green;

                        return (
                            <div key={key} style={{
                                background: 'rgba(255,255,255,0.05)',
                                borderRadius: '12px',
                                padding: '1.25rem',
                                borderLeft: `4px solid ${color}`
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                                    <span style={{ fontWeight: '600', textTransform: 'capitalize' }}>
                                        {parts[0].replace(/_/g, ' ')} → {parts[1].replace(/_/g, ' ')}
                                    </span>
                                    <span style={{
                                        background: `${color}20`,
                                        color: color,
                                        padding: '0.2rem 0.5rem',
                                        borderRadius: '4px',
                                        fontSize: '0.75rem',
                                        fontWeight: '600'
                                    }}>
                                        r = {val.best_correlation?.toFixed(2)}
                                    </span>
                                </div>
                                <div style={{ fontSize: '0.9rem' }}>
                                    {val.optimal_lag === 0 ? (
                                        <span style={{ color: COLORS.green }}>⚡ Move simultaneously</span>
                                    ) : val.optimal_lag < 0 ? (
                                        <span style={{ color: COLORS.blue }}>📊 {parts[0].replace(/_/g, ' ')} LEADS by {Math.abs(val.optimal_lag)} months</span>
                                    ) : (
                                        <span style={{ color: COLORS.purple }}>📊 {parts[1].replace(/_/g, ' ')} LEADS by {val.optimal_lag} months</span>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>

                <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(59, 130, 246, 0.15)', borderRadius: '10px' }}>
                    <strong>💡 Key Insight:</strong> Fed rate changes predict copper prices <strong>12 months</strong> in advance.
                    When the Fed cuts rates, expect infrastructure investment (and AI build-out) to accelerate.
                </div>
            </div>

            {/* Constraint Acceleration */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
                <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem' }}>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>📉 Constraint Acceleration</h2>
                    <div style={{ display: 'grid', gap: '1rem' }}>
                        {constraintData.map(item => {
                            const color = item.growth > 10 ? COLORS.red : item.growth > 0 ? COLORS.yellow : COLORS.green;
                            const icon = item.growth > 10 ? '⚠️' : item.growth > 0 ? '🟡' : '✅';

                            return (
                                <div key={item.name} style={{
                                    background: 'rgba(255,255,255,0.05)',
                                    borderRadius: '10px',
                                    padding: '1rem',
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center'
                                }}>
                                    <div>
                                        <div style={{ fontWeight: '600' }}>{item.name}</div>
                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                                            Trend: {item.trend}
                                        </div>
                                    </div>
                                    <div style={{ textAlign: 'right' }}>
                                        <div style={{ fontSize: '1.25rem', fontWeight: '700', color }}>
                                            {item.growth > 0 ? '+' : ''}{item.growth.toFixed(1)}%
                                        </div>
                                        <div style={{ fontSize: '0.75rem' }}>{icon} per year</div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Infrastructure Rate */}
                <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem' }}>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>🏗️ Infrastructure Build-Out Rate</h2>
                    <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.85rem' }}>
                        Copper/Industrial Production ratio as proxy for data center demand
                    </p>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                        <div style={{ background: 'rgba(59, 130, 246, 0.15)', padding: '1rem', borderRadius: '10px', textAlign: 'center' }}>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: COLORS.blue }}>{dd5.pre_2020_ratio}</div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Pre-2020</div>
                        </div>
                        <div style={{ background: 'rgba(139, 92, 246, 0.15)', padding: '1rem', borderRadius: '10px', textAlign: 'center' }}>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: COLORS.purple }}>{dd5['2023_plus_ratio']}</div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>2023+</div>
                        </div>
                    </div>

                    <div style={{
                        padding: '1rem',
                        background: 'rgba(245, 158, 11, 0.15)',
                        borderRadius: '10px',
                        borderLeft: '4px solid #f59e0b'
                    }}>
                        <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>🟡 Elevated Infrastructure Demand</div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                            Infrastructure intensity has increased +85% since pre-pandemic.
                            This signals accelerating data center and AI infrastructure build-out.
                        </div>
                    </div>
                </div>
            </div>

            {/* 12-Month Forecast */}
            <div className="card" style={{ background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%)', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>🔮 12-Month Copper Forecast (LSTM Model)</h2>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.25rem', borderRadius: '12px', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Current Price</div>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>${dd6.current_price?.toLocaleString()}</div>
                    </div>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.25rem', borderRadius: '12px', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>12-Month Forecast</div>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700', color: COLORS.cyan }}>${parseInt(dd6.forecast_12m).toLocaleString()}</div>
                    </div>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.25rem', borderRadius: '12px', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Predicted Change</div>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700', color: dd6.forecast_change_pct < 0 ? COLORS.green : COLORS.red }}>
                            {dd6.forecast_change_pct}%
                        </div>
                    </div>
                </div>

                <div style={{
                    padding: '1rem',
                    background: 'rgba(16, 185, 129, 0.2)',
                    borderRadius: '10px',
                    borderLeft: '4px solid #10b981'
                }}>
                    <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>✅ No Immediate Constraint</div>
                    <div style={{ fontSize: '0.9rem' }}>
                        LSTM model forecasts copper prices to ease over the next 12 months,
                        remaining within historical range. AI infrastructure can continue scaling without material bottlenecks.
                    </div>
                </div>
            </div>

            {/* Multi-Metal Analysis Section */}
            <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '0.5rem' }}>🔩 AI Infrastructure Metals Analysis</h2>
                <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem', fontSize: '0.85rem' }}>
                    All critical metals for AI data centers: wiring, cooling, circuit boards, batteries
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
                    {Object.entries(metalsData.metals_analysis || {}).map(([metal, data]) => {
                        const changePct = data.chatgpt_era_change_pct;
                        const color = changePct > 80 ? COLORS.red : changePct > 40 ? COLORS.yellow : COLORS.green;
                        const trend = data.recent_12m_trend;
                        const trendColor = trend === 'rising' ? COLORS.red : COLORS.green;

                        return (
                            <div key={metal} style={{
                                background: 'rgba(255,255,255,0.05)',
                                borderRadius: '12px',
                                padding: '1.25rem',
                                borderLeft: `4px solid ${color}`
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                                    <div>
                                        <div style={{ fontWeight: '600', fontSize: '1.1rem', textTransform: 'capitalize' }}>{metal}</div>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{data.use_case}</div>
                                    </div>
                                    <div style={{
                                        background: `${color}20`,
                                        color: color,
                                        padding: '0.25rem 0.5rem',
                                        borderRadius: '6px',
                                        fontSize: '0.85rem',
                                        fontWeight: '700'
                                    }}>
                                        +{changePct}%
                                    </div>
                                </div>

                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.85rem' }}>
                                    <div>
                                        <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Current</div>
                                        <div style={{ fontWeight: '600' }}>${data.current_price?.toLocaleString()}</div>
                                    </div>
                                    <div>
                                        <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>vs ATH</div>
                                        <div style={{ fontWeight: '600' }}>{data.pct_from_ath}%</div>
                                    </div>
                                    <div style={{ gridColumn: 'span 2' }}>
                                        <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>12-month trend</div>
                                        <div style={{ fontWeight: '600', color: trendColor, textTransform: 'capitalize' }}>
                                            {trend} ({data.recent_12m_change_pct > 0 ? '+' : ''}{data.recent_12m_change_pct}%)
                                        </div>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* AI Infrastructure Index */}
                <div style={{
                    marginTop: '1.5rem',
                    padding: '1.25rem',
                    background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(99, 102, 241, 0.1) 100%)',
                    borderRadius: '12px'
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                        <div>
                            <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>🔧 AI Infrastructure Metals Index</div>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                                Weighted composite of critical AI metals (Copper 35%, Aluminum 25%, Tin 20%, Nickel 15%)
                            </div>
                        </div>
                        <div style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: COLORS.yellow }}>
                                +{metalsData.ai_infrastructure_index?.current_value?.toFixed(1)}σ
                            </div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                {metalsData.ai_infrastructure_index?.status?.replace(/[🔴🟡🟢✅]/g, '')}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Key metals findings */}
                <div style={{ marginTop: '1rem', display: 'grid', gap: '0.5rem' }}>
                    {(metalsData.key_findings || []).slice(0, 4).map((finding, i) => (
                        <div key={i} style={{
                            background: 'rgba(255,255,255,0.03)',
                            padding: '0.75rem 1rem',
                            borderRadius: '8px',
                            fontSize: '0.9rem'
                        }}>
                            {finding}
                        </div>
                    ))}
                </div>
            </div>

            {/* Bottom Line */}
            <div style={{
                background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 100%)',
                borderRadius: '16px',
                padding: '2rem',
                textAlign: 'center'
            }}>
                <h2 style={{ fontSize: '1.5rem', fontWeight: '700', marginBottom: '1rem' }}>🎯 Bottom Line: AI Timeline Outlook</h2>
                <p style={{ fontSize: '1.1rem', maxWidth: '700px', margin: '0 auto 1.5rem', lineHeight: 1.6 }}>
                    Physical infrastructure is <strong>not yet the binding constraint</strong> on AI development.
                    However, copper at <strong>96% of all-time high</strong> and <strong>elevated infrastructure intensity</strong>
                    signal that supply chains are under pressure.
                </p>
                <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', flexWrap: 'wrap' }}>
                    <div>
                        <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Energy</div>
                        <div style={{ fontSize: '1.25rem', fontWeight: '600', color: COLORS.green }}>✅ No Constraint</div>
                    </div>
                    <div>
                        <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Copper/Materials</div>
                        <div style={{ fontSize: '1.25rem', fontWeight: '600', color: COLORS.yellow }}>🟡 Elevated</div>
                    </div>
                    <div>
                        <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Capital Costs</div>
                        <div style={{ fontSize: '1.25rem', fontWeight: '600', color: COLORS.yellow }}>🟡 High Rates</div>
                    </div>
                </div>
            </div>

            {/* Footer */}
            <div style={{ marginTop: '2rem', padding: '1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Generated: {new Date(deepDiveData.generated_at).toLocaleString()} •
                Data: FRED, World Bank •
                Models: LSTM (PyTorch), Cross-correlation analysis •
                GPU: NVIDIA RTX 3060
            </div>
        </div>
    );
}
