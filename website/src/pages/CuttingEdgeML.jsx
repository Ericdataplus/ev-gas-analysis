import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, RadarChart, PolarGrid, PolarAngleAxis, Radar, PieChart, Pie, Legend } from 'recharts'
import analysisData from '../data/cutting_edge_ml.json'

const COLORS = ['#6366f1', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#14b8a6', '#eab308', '#84cc16']

export default function CuttingEdgeML() {
    const [activeTab, setActiveTab] = useState('overview')
    const models = analysisData?.models || []

    const tabs = [
        { id: 'overview', label: '📊 Overview', color: '#6366f1' },
        { id: 'deep', label: '🧠 Deep Learning', color: '#22c55e' },
        { id: 'optimization', label: '🧬 Optimization', color: '#f97316' },
        { id: 'analysis', label: '🔍 Analysis', color: '#8b5cf6' },
    ]

    const lstm = models.find(m => m.model === 'LSTM') || {}
    const transformer = models.find(m => m.model === 'Transformer') || {}
    const autoencoder = models.find(m => m.model === 'Autoencoder') || {}
    const multitask = models.find(m => m.model === 'MultiTask') || {}
    const genetic = models.find(m => m.model === 'GeneticAlgorithm') || {}
    const rl = models.find(m => m.model === 'QLearning') || {}
    const survival = models.find(m => m.model === 'SurvivalAnalysis') || {}
    const graph = models.find(m => m.model === 'GraphAnalysis') || {}
    const counterfactual = models.find(m => m.model === 'Counterfactual') || {}
    const importance = models.find(m => m.model === 'FeatureImportance') || {}

    const featureData = importance.features?.map((f, i) => ({
        name: f.name.replace('_', ' '),
        value: Math.round(f.importance * 100)
    })) || []

    const supplierData = graph.critical_suppliers?.map((s, i) => ({
        name: s.name.replace('_', ' '),
        centrality: Math.round(s.centrality * 100)
    })) || []

    const scenarioData = counterfactual.scenarios?.map(s => ({
        name: s.scenario.replace(/_/g, ' '),
        impact: s.impact_pct
    })) || []

    return (
        <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
            {/* Header */}
            <div style={{
                background: 'linear-gradient(135deg, #1e1b4b 0%, #4c1d95 50%, #7c3aed 100%)',
                borderRadius: '20px', padding: '2.5rem', marginBottom: '2rem', color: 'white',
                position: 'relative', overflow: 'hidden'
            }}>
                <div style={{ position: 'absolute', top: -30, right: -30, fontSize: '15rem', opacity: 0.1 }}>🧠</div>
                <div style={{ fontSize: '0.8rem', background: 'rgba(239, 68, 68, 0.4)', display: 'inline-block', padding: '0.25rem 0.75rem', borderRadius: '20px', marginBottom: '1rem', fontWeight: '600' }}>
                    🔬 Research-Level Machine Learning
                </div>
                <h1 style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>🧠 Cutting-Edge ML Suite</h1>
                <p style={{ fontSize: '1.1rem', opacity: 0.9, maxWidth: '700px' }}>
                    10 advanced techniques: LSTM, Transformer, Autoencoder, Multi-Task Learning,
                    Genetic Algorithms, Reinforcement Learning, Survival Analysis, and more.
                </p>
                <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>10</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>ML Techniques</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>{Math.round((multitask.adoption_r2 || 0) * 100)}%</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Best R² Score</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>{analysisData?.execution_seconds || 0}s</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Runtime</div>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
                {tabs.map(tab => (
                    <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
                        padding: '0.75rem 1.5rem', borderRadius: '10px', border: 'none',
                        background: activeTab === tab.id ? `linear-gradient(135deg, ${tab.color}, ${tab.color}dd)` : 'var(--bg-card)',
                        color: activeTab === tab.id ? 'white' : 'var(--text-secondary)',
                        fontWeight: '600', cursor: 'pointer'
                    }}>{tab.label}</button>
                ))}
            </div>

            {/* OVERVIEW */}
            {activeTab === 'overview' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>📊 All Models Overview</h2>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
                        {models.map((m, i) => (
                            <div key={i} className="card" style={{ padding: '1.25rem', borderLeft: `4px solid ${COLORS[i]}` }}>
                                <h4 style={{ marginBottom: '0.5rem' }}>{m.model}</h4>
                                {m.r2 && <div style={{ fontSize: '1.5rem', fontWeight: '700', color: COLORS[i] }}>R² = {(m.r2 * 100).toFixed(1)}%</div>}
                                {m.anomalies && <div style={{ fontSize: '1.5rem', fontWeight: '700', color: COLORS[i] }}>{m.anomalies} anomalies</div>}
                                {m.optimal_price && <div style={{ fontSize: '1.5rem', fontWeight: '700', color: COLORS[i] }}>${m.optimal_price.toLocaleString()}</div>}
                                {m.scenarios && <div style={{ fontSize: '1.5rem', fontWeight: '700', color: COLORS[i] }}>{m.scenarios.length} scenarios</div>}
                                {m.features && <div style={{ fontSize: '1.5rem', fontWeight: '700', color: COLORS[i] }}>{m.features.length} features ranked</div>}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* DEEP LEARNING */}
            {activeTab === 'deep' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🧠 Deep Learning Models</h2>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
                        <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #6366f1' }}>
                            <h3>📈 LSTM</h3>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Long Short-Term Memory for time series</p>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#6366f1' }}>{lstm.layers || 2} layers, {lstm.hidden || 64} hidden</div>
                        </div>
                        <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #22c55e' }}>
                            <h3>🔮 Transformer</h3>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Attention-based sequence modeling</p>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#22c55e' }}>{transformer.heads || 4} attention heads</div>
                        </div>
                        <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #f97316' }}>
                            <h3>🔍 Autoencoder</h3>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Anomaly detection via reconstruction</p>
                            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#f97316' }}>{autoencoder.anomalies || 0} anomalies detected</div>
                        </div>
                        <div className="card" style={{ padding: '1.5rem', borderTop: '4px solid #8b5cf6' }}>
                            <h3>🎯 Multi-Task</h3>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Shared representation learning</p>
                            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                                <span style={{ background: '#22c55e20', color: '#22c55e', padding: '0.25rem 0.5rem', borderRadius: '6px', fontSize: '0.8rem' }}>
                                    Adoption: {((multitask.adoption_r2 || 0) * 100).toFixed(0)}%
                                </span>
                                <span style={{ background: '#6366f120', color: '#6366f1', padding: '0.25rem 0.5rem', borderRadius: '6px', fontSize: '0.8rem' }}>
                                    Battery: {((multitask.battery_r2 || 0) * 100).toFixed(0)}%
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* OPTIMIZATION */}
            {activeTab === 'optimization' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🧬 Optimization & RL</h2>
                    <div className="grid-2">
                        <div className="card" style={{ padding: '1.5rem' }}>
                            <h3 style={{ marginBottom: '1rem' }}>🧬 Genetic Algorithm</h3>
                            <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>Found optimal EV configuration:</p>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                                <div style={{ padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Price</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: '700' }}>${(genetic.optimal_price || 0).toLocaleString()}</div>
                                </div>
                                <div style={{ padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Range</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: '700' }}>{genetic.optimal_range || 0} mi</div>
                                </div>
                                <div style={{ padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Charging</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: '700' }}>{genetic.optimal_charging_speed || 0} kW</div>
                                </div>
                                <div style={{ padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Battery Cost</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: '700' }}>${genetic.optimal_battery_cost || 0}/kWh</div>
                                </div>
                            </div>
                        </div>
                        <div className="card" style={{ padding: '1.5rem' }}>
                            <h3 style={{ marginBottom: '1rem' }}>🎮 Reinforcement Learning</h3>
                            <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>Q-Learning for charging optimization</p>
                            <div style={{ padding: '1rem', background: 'rgba(34, 197, 94, 0.1)', borderRadius: '8px', borderLeft: '3px solid #22c55e' }}>
                                <strong>Optimal Charging:</strong> Off-peak hours (2-5 AM) when electricity is cheapest
                            </div>
                            <div style={{ marginTop: '0.75rem', fontSize: '0.9rem' }}>
                                <strong>Episodes trained:</strong> {rl.episodes?.toLocaleString() || 5000}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* ANALYSIS */}
            {activeTab === 'analysis' && (
                <div>
                    <h2 style={{ marginBottom: '1.5rem' }}>🔍 Advanced Analysis</h2>

                    {/* Feature Importance */}
                    {featureData.length > 0 && (
                        <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                            <h3 className="chart-title">Feature Importance (%)</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={featureData} layout="vertical">
                                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                    <XAxis type="number" stroke="#71717a" />
                                    <YAxis dataKey="name" type="category" width={120} stroke="#71717a" />
                                    <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} />
                                    <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                                        {featureData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Counterfactual Scenarios */}
                    {scenarioData.length > 0 && (
                        <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                            <h3 className="chart-title">Policy Counterfactual Impact (%)</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={scenarioData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                    <XAxis dataKey="name" stroke="#71717a" angle={-20} textAnchor="end" height={80} fontSize={10} />
                                    <YAxis stroke="#71717a" />
                                    <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a' }} />
                                    <Bar dataKey="impact" radius={[8, 8, 0, 0]}>
                                        {scenarioData.map((entry, i) => (
                                            <Cell key={i} fill={entry.impact > 0 ? '#22c55e' : '#ef4444'} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Supply Chain & Survival */}
                    <div className="grid-2">
                        <div className="card" style={{ padding: '1.5rem' }}>
                            <h4 style={{ marginBottom: '1rem' }}>🔗 Supply Chain Criticality</h4>
                            {supplierData.map((s, i) => (
                                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-color)' }}>
                                    <span>{s.name}</span>
                                    <span style={{ fontWeight: '600', color: COLORS[i] }}>{s.centrality}%</span>
                                </div>
                            ))}
                        </div>
                        <div className="card" style={{ padding: '1.5rem' }}>
                            <h4 style={{ marginBottom: '1rem' }}>⏳ Battery Survival Analysis</h4>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.5rem' }}>
                                <div style={{ textAlign: 'center', padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>10% Fail</div>
                                    <div style={{ fontWeight: '700' }}>{survival['10%_fail'] || 0}</div>
                                </div>
                                <div style={{ textAlign: 'center', padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>50% Fail</div>
                                    <div style={{ fontWeight: '700' }}>{survival['50%_fail'] || 0}</div>
                                </div>
                                <div style={{ textAlign: 'center', padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>90% Fail</div>
                                    <div style={{ fontWeight: '700' }}>{survival['90%_fail'] || 0}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
