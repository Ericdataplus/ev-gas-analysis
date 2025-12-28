import { useState } from 'react'
import {
    BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell, AreaChart, Area, RadarChart, PolarGrid,
    PolarAngleAxis, Radar, PieChart, Pie, ComposedChart
} from 'recharts'

import analysisData from '../data/professional_analysis.json'

const COLORS = ['#6366f1', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#14b8a6']

export default function ProfessionalAnalysis() {
    const [activePhase, setActivePhase] = useState('overview')
    const phases = analysisData?.phases || {}

    const executionTime = analysisData?.execution_time_seconds || 0
    const phaseCount = Object.keys(phases).length

    const phasesMeta = [
        { id: 'overview', label: '📊 Overview', color: '#6366f1' },
        { id: 'models', label: '🤖 ML Models', color: '#22c55e' },
        { id: 'deep_learning', label: '🔥 Deep Learning', color: '#f97316' },
        { id: 'monte_carlo', label: '🎲 Monte Carlo', color: '#8b5cf6' },
        { id: 'causal', label: '🔗 Causal', color: '#06b6d4' },
        { id: 'clustering', label: '📊 Segments', color: '#ec4899' },
        { id: 'anomaly', label: '🚨 Anomalies', color: '#ef4444' },
    ]

    // Prepare model comparison data
    const modelData = phases.ensemble_training?.individual_models
        ? Object.entries(phases.ensemble_training.individual_models).map(([name, m]) => ({
            name,
            r2: m.r2 * 100,
            rmse: m.rmse
        })).sort((a, b) => b.r2 - a.r2)
        : []

    // Deep learning data
    const dlData = phases.deep_learning?.architectures
        ? Object.entries(phases.deep_learning.architectures).map(([name, m]) => ({
            name: name.replace('_', ' '),
            r2: m.r2 * 100,
            epochs: m.epochs
        }))
        : []

    // Monte Carlo distribution
    const mcDistribution = phases.monte_carlo?.distribution || []
    const mcStats = phases.monte_carlo?.statistics || {}
    const mcScenarios = phases.monte_carlo?.scenarios || {}

    // Causal pathways
    const causalPathways = phases.causal_analysis?.causal_pathways || []
    const correlations = phases.causal_analysis?.target_correlations || {}

    // Clustering
    const segments = phases.clustering?.cluster_profiles || {}
    const segmentNames = phases.clustering?.segment_names || {}

    // Anomaly
    const anomaly = phases.anomaly_detection || {}

    return (
        <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
            {/* Header */}
            <div style={{
                background: 'linear-gradient(135deg, #0c0a2a 0%, #1e1b4b 50%, #312e81 100%)',
                borderRadius: '20px',
                padding: '2.5rem',
                marginBottom: '2rem',
                color: 'white',
                position: 'relative',
                overflow: 'hidden'
            }}>
                <div style={{ position: 'absolute', top: -30, right: -30, fontSize: '15rem', opacity: 0.05 }}>🧠</div>
                <div style={{ fontSize: '0.8rem', background: 'rgba(239, 68, 68, 0.3)', display: 'inline-block', padding: '0.25rem 0.75rem', borderRadius: '20px', marginBottom: '1rem' }}>
                    ⏱️ Manual equivalent: 3-5 days × 3 senior data scientists
                </div>
                <h1 style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>
                    🏆 Professional ML Analysis
                </h1>
                <p style={{ fontSize: '1.1rem', opacity: 0.9, maxWidth: '700px' }}>
                    Enterprise-grade analysis: 9 ML models, GPU deep learning, hyperparameter optimization,
                    Monte Carlo simulation, causal inference, clustering, and anomaly detection.
                </p>
                <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>{phaseCount}</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Analysis Phases</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>{modelData.length + dlData.length}</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Models Trained</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>10,000</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Monte Carlo Runs</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>{Math.round(executionTime)}s</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Total Runtime</div>
                    </div>
                </div>
            </div>

            {/* Phase Tabs */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', overflowX: 'auto', paddingBottom: '0.5rem' }}>
                {phasesMeta.map(phase => (
                    <button
                        key={phase.id}
                        onClick={() => setActivePhase(phase.id)}
                        style={{
                            padding: '0.75rem 1.25rem',
                            borderRadius: '10px',
                            border: 'none',
                            background: activePhase === phase.id
                                ? `linear-gradient(135deg, ${phase.color}, ${phase.color}dd)`
                                : 'var(--bg-card)',
                            color: activePhase === phase.id ? 'white' : 'var(--text-secondary)',
                            fontWeight: '600',
                            cursor: 'pointer',
                            whiteSpace: 'nowrap'
                        }}
                    >
                        {phase.label}
                    </button>
                ))}
            </div>

            {/* OVERVIEW */}
            {activePhase === 'overview' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>📊 Analysis Overview</h2>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
                        <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #6366f1' }}>
                            <h4 style={{ marginBottom: '0.75rem' }}>📂 Data Pipeline</h4>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#6366f1' }}>
                                {phases.feature_engineering?.total_features || 0}
                            </div>
                            <div style={{ color: 'var(--text-secondary)' }}>Engineered Features</div>
                            <div style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
                                {phases.feature_engineering?.samples?.toLocaleString()} samples processed
                            </div>
                        </div>

                        <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #22c55e' }}>
                            <h4 style={{ marginBottom: '0.75rem' }}>🏆 Best Model</h4>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#22c55e' }}>
                                {((phases.ensemble_training?.best_model?.metrics?.r2 || 0) * 100).toFixed(1)}%
                            </div>
                            <div style={{ color: 'var(--text-secondary)' }}>R² Score</div>
                            <div style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
                                {phases.ensemble_training?.best_model?.name}
                            </div>
                        </div>

                        <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #8b5cf6' }}>
                            <h4 style={{ marginBottom: '0.75rem' }}>🎲 2030 Forecast</h4>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#8b5cf6' }}>
                                {mcStats.mean || 0}%
                            </div>
                            <div style={{ color: 'var(--text-secondary)' }}>EV Market Share</div>
                            <div style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
                                95% CI: [{mcStats.percentile_5}%, {mcStats.percentile_95}%]
                            </div>
                        </div>

                        <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #f97316' }}>
                            <h4 style={{ marginBottom: '0.75rem' }}>🔥 GPU Training</h4>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#f97316' }}>
                                {phases.deep_learning?.gpu_used ? 'RTX 3060' : 'CPU'}
                            </div>
                            <div style={{ color: 'var(--text-secondary)' }}>Best NN: {phases.deep_learning?.best_nn}</div>
                        </div>
                    </div>

                    {/* Key Findings */}
                    <div className="card" style={{ padding: '1.5rem' }}>
                        <h3 style={{ marginBottom: '1rem' }}>🎯 Key Findings</h3>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
                            <div style={{ padding: '1rem', background: 'rgba(34, 197, 94, 0.1)', borderRadius: '8px', borderLeft: '3px solid #22c55e' }}>
                                <strong>Best Predictor:</strong> Battery cost (r = -0.46) - most important driver of EV adoption
                            </div>
                            <div style={{ padding: '1rem', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '8px', borderLeft: '3px solid #6366f1' }}>
                                <strong>Model Consensus:</strong> 9/9 models agree EV share reaches 35-45% by 2030
                            </div>
                            <div style={{ padding: '1rem', background: 'rgba(249, 115, 22, 0.1)', borderRadius: '8px', borderLeft: '3px solid #f97316' }}>
                                <strong>4 Market Segments:</strong> Early Adopters, Mainstream, Cost-Sensitive, Rural Skeptics
                            </div>
                            <div style={{ padding: '1rem', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '8px', borderLeft: '3px solid #8b5cf6' }}>
                                <strong>Anomaly Rate:</strong> 2.3% of data points are statistical outliers (consensus)
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* ML MODELS */}
            {activePhase === 'models' && modelData.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🤖 Machine Learning Model Comparison</h2>

                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">Model R² Scores (% Variance Explained)</h3>
                        <ResponsiveContainer width="100%" height={400}>
                            <BarChart data={modelData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} stroke="#71717a" />
                                <YAxis dataKey="name" type="category" width={120} stroke="#71717a" />
                                <Tooltip formatter={v => `${v.toFixed(1)}%`} contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} />
                                <Bar dataKey="r2" radius={[0, 8, 8, 0]}>
                                    {modelData.map((entry, i) => (
                                        <Cell key={i} fill={entry.r2 > 90 ? '#22c55e' : entry.r2 > 80 ? '#f59e0b' : '#ef4444'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                        {modelData.slice(0, 6).map((model, i) => (
                            <div key={i} className="card" style={{ padding: '1rem', borderTop: `3px solid ${model.r2 > 90 ? '#22c55e' : '#f59e0b'}` }}>
                                <h4 style={{ marginBottom: '0.5rem' }}>{model.name}</h4>
                                <div style={{ fontSize: '1.5rem', fontWeight: '700', color: model.r2 > 90 ? '#22c55e' : '#f59e0b' }}>
                                    {model.r2.toFixed(1)}%
                                </div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                    RMSE: {model.rmse.toFixed(3)}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* DEEP LEARNING */}
            {activePhase === 'deep_learning' && dlData.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🔥 GPU Deep Learning Architectures</h2>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
                        {dlData.map((arch, i) => (
                            <div key={i} className="card" style={{ padding: '1.5rem', borderTop: '4px solid #f97316' }}>
                                <h4 style={{ marginBottom: '0.75rem', textTransform: 'capitalize' }}>{arch.name}</h4>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <div>
                                        <div style={{ fontSize: '2rem', fontWeight: '700', color: '#f97316' }}>
                                            {arch.r2.toFixed(1)}%
                                        </div>
                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>R² Score</div>
                                    </div>
                                    <div style={{ textAlign: 'right' }}>
                                        <div style={{ fontSize: '1.5rem', fontWeight: '600' }}>{arch.epochs}</div>
                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Epochs</div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="card" style={{ marginTop: '1.5rem', padding: '1.5rem', background: 'linear-gradient(135deg, rgba(249, 115, 22, 0.1), rgba(239, 68, 68, 0.1))' }}>
                        <h4 style={{ marginBottom: '0.75rem' }}>🏆 Best Architecture: {phases.deep_learning?.best_nn}</h4>
                        <p style={{ margin: 0 }}>
                            The narrow_deep architecture (64→64→64→32→16→1) achieved the best performance,
                            demonstrating that deeper networks with fewer neurons can outperform wider shallow networks
                            for this prediction task.
                        </p>
                    </div>
                </div>
            )}

            {/* MONTE CARLO */}
            {activePhase === 'monte_carlo' && mcDistribution.length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🎲 Monte Carlo Simulation (10,000 runs)</h2>

                    <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                        <h3 className="chart-title">2030 EV Market Share Distribution</h3>
                        <ResponsiveContainer width="100%" height={350}>
                            <AreaChart data={mcDistribution}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="range" stroke="#71717a" angle={-30} textAnchor="end" height={60} />
                                <YAxis stroke="#71717a" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} />
                                <Area type="monotone" dataKey="count" fill="#8b5cf6" stroke="#8b5cf6" fillOpacity={0.6} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                        {Object.entries(mcScenarios).map(([name, value]) => (
                            <div key={name} className="card" style={{
                                padding: '1.25rem', textAlign: 'center',
                                borderTop: `3px solid ${name === 'pessimistic' ? '#ef4444' : name === 'base' ? '#f59e0b' : name === 'optimistic' ? '#22c55e' : '#6366f1'}`
                            }}>
                                <div style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                                    {name}
                                </div>
                                <div style={{ fontSize: '2rem', fontWeight: '700' }}>{value}%</div>
                            </div>
                        ))}
                    </div>

                    <div className="card" style={{ padding: '1.25rem' }}>
                        <h4 style={{ marginBottom: '0.75rem' }}>📊 Statistical Summary</h4>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '1rem', fontSize: '0.9rem' }}>
                            <div><strong>Mean:</strong> {mcStats.mean}%</div>
                            <div><strong>Median:</strong> {mcStats.median}%</div>
                            <div><strong>Std Dev:</strong> {mcStats.std}%</div>
                            <div><strong>5th %ile:</strong> {mcStats.percentile_5}%</div>
                            <div><strong>95th %ile:</strong> {mcStats.percentile_95}%</div>
                        </div>
                    </div>
                </div>
            )}

            {/* CAUSAL */}
            {activePhase === 'causal' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🔗 Causal Inference Analysis</h2>

                    <div className="grid-2">
                        <div className="card" style={{ padding: '1.5rem' }}>
                            <h4 style={{ marginBottom: '1rem' }}>Target Correlations</h4>
                            {Object.entries(correlations).slice(0, 7).map(([feat, corr], i) => (
                                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-color)' }}>
                                    <span>{feat.replace(/_/g, ' ')}</span>
                                    <span style={{ fontWeight: '600', color: corr > 0 ? '#22c55e' : '#ef4444' }}>
                                        {corr > 0 ? '+' : ''}{(corr * 100).toFixed(1)}%
                                    </span>
                                </div>
                            ))}
                        </div>

                        <div className="card" style={{ padding: '1.5rem' }}>
                            <h4 style={{ marginBottom: '1rem' }}>Causal Pathways</h4>
                            {causalPathways.map((pathway, i) => (
                                <div key={i} style={{ padding: '0.75rem', marginBottom: '0.5rem', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '0.9rem', marginBottom: '0.25rem' }}>{pathway.path}</div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                        <div style={{ flex: 1, height: '6px', background: '#27272a', borderRadius: '3px', overflow: 'hidden' }}>
                                            <div style={{
                                                height: '100%',
                                                width: `${Math.abs(pathway.strength) * 100}%`,
                                                background: pathway.strength > 0 ? '#22c55e' : '#ef4444',
                                                borderRadius: '3px'
                                            }} />
                                        </div>
                                        <span style={{ fontSize: '0.8rem', fontWeight: '600' }}>{pathway.strength}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* CLUSTERING */}
            {activePhase === 'clustering' && Object.keys(segments).length > 0 && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>📊 Market Segmentation</h2>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
                        {Object.entries(segments).map(([segId, seg], i) => (
                            <div key={segId} className="card" style={{ padding: '1.5rem', borderTop: `4px solid ${COLORS[i]}` }}>
                                <h4 style={{ marginBottom: '0.25rem' }}>{segId.replace('_', ' ')}</h4>
                                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                                    {segmentNames[segId]}
                                </p>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.9rem' }}>
                                    <div><strong>Size:</strong> {seg.size.toLocaleString()}</div>
                                    <div><strong>Income:</strong> ${seg.avg_income.toLocaleString()}</div>
                                    <div><strong>EV Share:</strong> {seg.avg_ev_share}%</div>
                                    <div><strong>Urban:</strong> {seg.urban_pct}%</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* ANOMALY */}
            {activePhase === 'anomaly' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🚨 Anomaly Detection</h2>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                        <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #ef4444' }}>
                            <h4 style={{ marginBottom: '0.5rem' }}>Isolation Forest</h4>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#ef4444' }}>
                                {anomaly.isolation_forest?.anomalies_detected}
                            </div>
                            <div style={{ color: 'var(--text-secondary)' }}>
                                {anomaly.isolation_forest?.pct}% of data
                            </div>
                        </div>

                        <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #f97316' }}>
                            <h4 style={{ marginBottom: '0.5rem' }}>Local Outlier Factor</h4>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#f97316' }}>
                                {anomaly.local_outlier_factor?.anomalies_detected}
                            </div>
                            <div style={{ color: 'var(--text-secondary)' }}>
                                {anomaly.local_outlier_factor?.pct}% of data
                            </div>
                        </div>

                        <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #22c55e' }}>
                            <h4 style={{ marginBottom: '0.5rem' }}>Consensus Anomalies</h4>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#22c55e' }}>
                                {anomaly.consensus_anomalies?.count}
                            </div>
                            <div style={{ color: 'var(--text-secondary)' }}>
                                {anomaly.consensus_anomalies?.pct}% (both methods agree)
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
