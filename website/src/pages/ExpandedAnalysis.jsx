import { useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    ScatterChart, Scatter, Cell, PieChart, Pie
} from 'recharts';
import analysisData from '../data/expanded_analysis_results.json';

const COLORS = {
    strong: '#10b981',
    moderate: '#f59e0b',
    weak: '#ef4444',
    primary: '#6366f1',
    secondary: '#8b5cf6'
};

export default function ExpandedAnalysis() {
    const [activeSection, setActiveSection] = useState('overview');

    // Prepare correlation data for chart
    const correlationData = analysisData.correlations.slice(0, 10).map(c => ({
        name: `${c.var1.split('_')[0]} ↔ ${c.var2.split('_')[0]}`,
        correlation: c.correlation,
        strength: c.strength,
        fullName: `${c.var1} ↔ ${c.var2}`
    }));

    // Prepare feature importance data
    const featureImportance = Object.entries(analysisData.ml_results?.copper_prediction?.feature_importance || {})
        .map(([name, value]) => ({ name: name.replace('_', ' '), value: value * 100 }))
        .sort((a, b) => b.value - a.value);

    // Anomaly dates
    const anomalyDates = analysisData.deep_learning?.autoencoder?.anomaly_dates || [];

    // Data sources
    const dataSources = [
        { name: 'FRED', count: analysisData.data_sources.fred.length, color: '#3b82f6' },
        { name: 'OWID', count: analysisData.data_sources.owid.length, color: '#10b981' },
        { name: 'Kaggle', count: analysisData.data_sources.kaggle.length, color: '#f59e0b' },
        { name: 'EIA', count: analysisData.data_sources.eia.length, color: '#ef4444' }
    ];

    return (
        <div className="dashboard">
            <div className="page-header" style={{ marginBottom: '2rem' }}>
                <h1 style={{ fontSize: '2rem', fontWeight: '700', marginBottom: '0.5rem' }}>
                    📊 Expanded ML Analysis
                </h1>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
                    Comprehensive cross-domain analysis using <strong>REAL downloaded data</strong> from
                    FRED, OWID, World Bank, Kaggle, and EIA
                </p>
            </div>

            {/* Stats Overview */}
            <div className="stats-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)', borderRadius: '12px', padding: '1.5rem', color: 'white' }}>
                    <div style={{ fontSize: '2.5rem', fontWeight: '700' }}>{analysisData.datasets_loaded}</div>
                    <div style={{ opacity: 0.9 }}>Datasets Loaded</div>
                </div>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', borderRadius: '12px', padding: '1.5rem', color: 'white' }}>
                    <div style={{ fontSize: '2.5rem', fontWeight: '700' }}>{analysisData.master_dataset_rows}</div>
                    <div style={{ opacity: 0.9 }}>Months of Data</div>
                </div>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)', borderRadius: '12px', padding: '1.5rem', color: 'white' }}>
                    <div style={{ fontSize: '2.5rem', fontWeight: '700' }}>{analysisData.correlations.length}</div>
                    <div style={{ opacity: 0.9 }}>Correlations Found</div>
                </div>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)', borderRadius: '12px', padding: '1.5rem', color: 'white' }}>
                    <div style={{ fontSize: '2.5rem', fontWeight: '700' }}>{(analysisData.ml_results?.copper_prediction?.r2_score * 100).toFixed(1)}%</div>
                    <div style={{ opacity: 0.9 }}>ML Accuracy (R²)</div>
                </div>
            </div>

            {/* Data Sources */}
            <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>📁 Real Data Sources</h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                    {Object.entries(analysisData.data_sources).map(([source, datasets]) => (
                        <div key={source} style={{
                            background: 'rgba(255,255,255,0.05)',
                            borderRadius: '12px',
                            padding: '1rem',
                            border: '1px solid rgba(255,255,255,0.1)'
                        }}>
                            <div style={{ fontWeight: '600', textTransform: 'uppercase', marginBottom: '0.5rem', color: source === 'fred' ? '#3b82f6' : source === 'owid' ? '#10b981' : source === 'kaggle' ? '#f59e0b' : '#ef4444' }}>
                                {source}
                            </div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                                {datasets.map(ds => (
                                    <span key={ds} style={{
                                        background: 'rgba(255,255,255,0.1)',
                                        padding: '0.25rem 0.5rem',
                                        borderRadius: '6px',
                                        fontSize: '0.8rem'
                                    }}>
                                        {ds}
                                    </span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Key Insights */}
            <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>💡 Key Insights from Real Data</h2>
                <div style={{ display: 'grid', gap: '0.75rem' }}>
                    {analysisData.insights.map((insight, i) => (
                        <div key={i} style={{
                            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
                            padding: '1rem',
                            borderRadius: '10px',
                            borderLeft: '4px solid #6366f1',
                            fontSize: '0.95rem'
                        }}>
                            {insight}
                        </div>
                    ))}
                </div>
            </div>

            {/* Cross-Domain Correlations */}
            <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>🔗 Cross-Domain Correlations</h2>
                <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
                    Statistically significant correlations (p &lt; 0.05) between commodities and economic indicators
                </p>
                <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={correlationData} layout="vertical" margin={{ left: 120 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis type="number" domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                        <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} />
                        <Tooltip
                            formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Correlation']}
                            contentStyle={{ background: '#1e1e2e', border: 'none', borderRadius: '8px' }}
                        />
                        <Bar dataKey="correlation" radius={[0, 8, 8, 0]}>
                            {correlationData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={
                                    entry.strength === 'Strong' ? COLORS.strong :
                                        entry.strength === 'Moderate' ? COLORS.moderate : COLORS.weak
                                } />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
                <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1rem', justifyContent: 'center' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: COLORS.strong }}></span>
                        Strong (r &gt; 0.7)
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: COLORS.moderate }}></span>
                        Moderate (0.5-0.7)
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: COLORS.weak }}></span>
                        Weak (&lt; 0.5)
                    </span>
                </div>
            </div>

            {/* ML Model Results */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
                {/* Feature Importance */}
                <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem' }}>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>🤖 ML Feature Importance</h2>
                    <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
                        GradientBoosting model: R² = {(analysisData.ml_results?.copper_prediction?.r2_score * 100).toFixed(1)}%
                    </p>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={featureImportance.slice(0, 6)} layout="vertical" margin={{ left: 100 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis type="number" tickFormatter={v => `${v.toFixed(0)}%`} />
                            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} />
                            <Tooltip
                                formatter={(value) => [`${value.toFixed(1)}%`, 'Importance']}
                                contentStyle={{ background: '#1e1e2e', border: 'none', borderRadius: '8px' }}
                            />
                            <Bar dataKey="value" fill="#6366f1" radius={[0, 8, 8, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Anomaly Detection */}
                <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem' }}>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>🔍 Anomaly Detection (GPU)</h2>
                    <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
                        Autoencoder detected {analysisData.deep_learning?.autoencoder?.anomaly_count} unusual market periods
                    </p>
                    <div style={{ display: 'grid', gap: '0.75rem' }}>
                        {anomalyDates.map((date, i) => {
                            const [year, month] = date.split('-');
                            const monthName = new Date(2000, parseInt(month) - 1, 1).toLocaleString('default', { month: 'short' });
                            return (
                                <div key={i} style={{
                                    background: 'rgba(239, 68, 68, 0.15)',
                                    padding: '0.75rem 1rem',
                                    borderRadius: '8px',
                                    borderLeft: '4px solid #ef4444',
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center'
                                }}>
                                    <span style={{ fontWeight: '500' }}>{monthName} {year}</span>
                                    <span style={{
                                        background: '#ef4444',
                                        padding: '0.2rem 0.5rem',
                                        borderRadius: '4px',
                                        fontSize: '0.75rem',
                                        color: 'white'
                                    }}>
                                        ⚠️ Anomaly
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                    <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '8px', fontSize: '0.85rem' }}>
                        <strong>Why these dates?</strong> 2011-02 (commodity spike), 2018 (trade war volatility),
                        2021-2022 (post-COVID recovery + Ukraine crisis)
                    </div>
                </div>
            </div>

            {/* Deep Learning Info */}
            <div className="card" style={{ background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)', borderRadius: '16px', padding: '1.5rem', marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>🧠 Deep Learning (RTX 3060 GPU)</h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '10px' }}>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>LSTM Sequences</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: '600' }}>{analysisData.deep_learning?.lstm?.sequences}</div>
                    </div>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '10px' }}>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>LSTM MAE</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: '600' }}>${analysisData.deep_learning?.lstm?.mae}</div>
                    </div>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '10px' }}>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Anomalies Found</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: '600' }}>{analysisData.deep_learning?.autoencoder?.anomaly_count}</div>
                    </div>
                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '10px' }}>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Time Range</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: '600' }}>2010-2025</div>
                    </div>
                </div>
            </div>

            {/* Country Clusters */}
            <div className="card" style={{ background: 'var(--card-bg)', borderRadius: '16px', padding: '1.5rem' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>🌍 Country GHG Emissions Clustering</h2>
                <p style={{ color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem' }}>
                    K-Means clustering of 195 countries based on greenhouse gas emissions patterns
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
                    {Object.entries(analysisData.clustering?.ghg_clusters || {}).map(([cluster, countries]) => (
                        <div key={cluster} style={{
                            background: cluster === '0' ? 'rgba(239, 68, 68, 0.1)' :
                                cluster === '1' ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                            padding: '1rem',
                            borderRadius: '10px',
                            borderLeft: `4px solid ${cluster === '0' ? '#ef4444' : cluster === '1' ? '#f59e0b' : '#10b981'}`
                        }}>
                            <div style={{ fontWeight: '600', marginBottom: '0.5rem' }}>
                                Cluster {parseInt(cluster) + 1}: {cluster === '0' ? 'High Emitters' : cluster === '1' ? 'Global' : 'Low/Medium Emitters'}
                            </div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                                {countries.slice(0, 8).map(country => (
                                    <span key={country} style={{
                                        background: 'rgba(255,255,255,0.1)',
                                        padding: '0.2rem 0.5rem',
                                        borderRadius: '4px',
                                        fontSize: '0.8rem'
                                    }}>
                                        {country}
                                    </span>
                                ))}
                                {countries.length > 8 && (
                                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                        +{countries.length - 8} more
                                    </span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Footer */}
            <div style={{ marginTop: '2rem', padding: '1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Generated: {new Date(analysisData.generated_at).toLocaleString()} •
                Data sources: FRED, OWID, World Bank, Kaggle, EIA •
                GPU: NVIDIA RTX 3060
            </div>
        </div>
    );
}
