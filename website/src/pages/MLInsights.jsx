import { useState, useEffect } from 'react'
import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ScatterChart, Scatter, ComposedChart } from 'recharts'
import ChartModal from '../components/ChartModal'

// Import both data sources
import mlInsightsBasic from '../data/ml_insights.json'
import comprehensiveResults from '../data/ml_comprehensive_results.json'

export default function MLInsights() {
    // Extract data from comprehensive results
    const metadata = comprehensiveResults.metadata || {}
    const correlations = comprehensiveResults.correlations || []
    const predictiveModels = comprehensiveResults.predictive_models || {}
    const yearForecasts = comprehensiveResults.year_forecasts || {}
    const lstmForecasts = comprehensiveResults.lstm_forecasts || {}
    const clustering = comprehensiveResults.clustering || {}
    const crossCorrelations = comprehensiveResults.cross_correlations || []
    const keyInsights = comprehensiveResults.key_insights || []

    // Model comparison data
    const modelComparisonData = Object.entries(predictiveModels).map(([name, data]) => ({
        name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        r2: Math.max(data.r2, -2), // Cap at -2 for display
        rmse: data.rmse,
        mae: data.mae
    })).sort((a, b) => b.r2 - a.r2)

    // Feature importance from best model
    const featureImportance = predictiveModels.random_forest?.feature_importance || {}
    const featureData = Object.entries(featureImportance)
        .map(([feature, importance]) => ({
            feature: feature.replace(/_/g, ' ').slice(0, 15),
            importance: importance
        }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 10)

    // Correlation data for chart
    const correlationChartData = correlations.slice(0, 12).map(c => ({
        pair: `${c.var1.slice(0, 8)} ↔ ${c.var2.slice(0, 8)}`,
        correlation: c.correlation,
        strength: c.strength
    }))

    // Year-by-year forecast data
    const copperForecasts = yearForecasts.copper_price || []
    const energyForecasts = yearForecasts.energy_index || []
    const lstmCopperForecasts = lstmForecasts.copper_price?.forecasts || []
    const lstmEnergyForecasts = lstmForecasts.energy_index?.forecasts || []

    // Combined forecast data for chart
    const forecastChartData = copperForecasts.map((cf, i) => ({
        year: cf.year,
        copper_gb: cf.prediction,
        copper_lower: cf.lower_bound,
        copper_upper: cf.upper_bound,
        copper_lstm: lstmCopperForecasts[i]?.prediction || null,
    }))

    // PCA data
    const pcaData = clustering.pca || {}
    const pcaVariance = pcaData.explained_variance || []
    const pcaCumulative = pcaData.cumulative_variance || []
    const pcaChartData = pcaVariance.map((v, i) => ({
        component: `PC${i + 1}`,
        variance: v * 100,
        cumulative: pcaCumulative[i] * 100
    }))

    // Anomaly dates
    const anomalyDates = clustering.anomaly_detection?.anomaly_dates || []

    // Regime data
    const regimeData = clustering.regime_detection || {}
    const regimeDist = regimeData.regime_distribution || {}
    const regimeChartData = Object.entries(regimeDist).map(([regime, count]) => ({
        regime,
        count,
        color: regime === 'Bull' ? '#22c55e' : regime === 'Bear' ? '#ef4444' : '#6b7280'
    }))

    // Cross correlation data
    const crossCorrData = crossCorrelations.slice(0, 8).map(cc => ({
        feature: cc.feature.replace(/_/g, ' ').slice(0, 12),
        target: cc.target,
        lag: cc.best_lag,
        correlation: cc.correlation,
        interpretation: cc.interpretation
    }))

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">🧠 Advanced ML Analytics</h1>
                <p className="page-subtitle">Comprehensive machine learning analysis: {metadata.data_points || 312} data points, {metadata.features?.length || 20} features</p>
            </header>

            {/* Key Stats */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon">📊</div>
                    <div className="stat-value">{correlations.length}+</div>
                    <div className="stat-label">Strong Correlations</div>
                    <div className="stat-change">r &gt; 0.7</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">🤖</div>
                    <div className="stat-value">{Object.keys(predictiveModels).length}</div>
                    <div className="stat-label">ML Models Trained</div>
                    <div className="stat-change">RF, GB, XGB, Ridge, Lasso, ENet</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">📈</div>
                    <div className="stat-value">{lstmForecasts.energy_index?.validation_r2 ? (lstmForecasts.energy_index.validation_r2 * 100).toFixed(0) : 47}%</div>
                    <div className="stat-label">LSTM Accuracy</div>
                    <div className="stat-change">Energy Index R²</div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon">⚠️</div>
                    <div className="stat-value">{clustering.anomaly_detection?.n_anomalies || 16}</div>
                    <div className="stat-label">Anomalies Detected</div>
                    <div className="stat-change">Isolation Forest</div>
                </div>
            </div>

            {/* Key Insights */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🎯 Key ML Discoveries</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    {keyInsights.map((insight, i) => (
                        <div key={i} className="card" style={{
                            padding: '1rem',
                            borderLeft: `3px solid ${insight.category === 'prediction' ? 'var(--accent-green)' :
                                insight.category === 'correlation' ? 'var(--accent-blue)' :
                                    insight.category === 'anomaly' ? 'var(--accent-orange)' :
                                        'var(--accent-purple)'
                                }`
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                <span style={{ fontSize: '1.2rem' }}>{insight.icon}</span>
                                <strong>{insight.title}</strong>
                            </div>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                                {insight.detail}
                            </p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Correlation Analysis */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📈 Top Correlations (30 Strong Relationships Found)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Correlation Strength"
                        insight="Fed Funds Rate and Treasury yields show the strongest correlation (r=0.959). Commodity prices (copper, tin, lead) strongly correlate with energy prices."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={correlationChartData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" domain={[0, 1]} stroke="#71717a" />
                                <YAxis dataKey="pair" type="category" stroke="#71717a" width={130} tick={{ fontSize: 11 }} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar dataKey="correlation" radius={[0, 8, 8, 0]}>
                                    {correlationChartData.map((entry, i) => (
                                        <Cell key={i} fill={entry.correlation > 0.9 ? '#22c55e' : entry.correlation > 0.8 ? '#3b82f6' : '#a855f7'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Top 5 Correlations</h4>
                        {correlations.slice(0, 5).map((corr, i) => (
                            <div key={i} className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ fontSize: '0.9rem' }}>
                                        {corr.var1.replace(/_/g, ' ')} ↔ {corr.var2.replace(/_/g, ' ')}
                                    </span>
                                    <span style={{
                                        color: corr.correlation > 0.9 ? 'var(--accent-green)' : 'var(--accent-blue)',
                                        fontWeight: 600
                                    }}>
                                        r = {corr.correlation.toFixed(3)}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Feature Importance */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🎯 Feature Importance (Random Forest)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="What Drives Copper Prices?"
                        insight="Tin, aluminum, and lead prices are the strongest predictors of copper. This reflects co-movement in commodity markets driven by global industrial demand."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={featureData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" />
                                <YAxis dataKey="feature" type="category" stroke="#71717a" width={100} tick={{ fontSize: 11 }} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar dataKey="importance" fill="#a855f7" radius={[0, 8, 8, 0]}>
                                    {featureData.map((entry, i) => (
                                        <Cell key={i} fill={i < 3 ? '#22c55e' : i < 6 ? '#3b82f6' : '#6b7280'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Feature Rankings</h4>
                        {featureData.slice(0, 5).map((feat, i) => (
                            <div key={i} className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                    <span>#{i + 1} {feat.feature}</span>
                                    <span style={{ color: 'var(--accent-purple)', fontWeight: 600 }}>
                                        {(feat.importance * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div style={{ background: '#27272a', borderRadius: '4px', height: '6px', overflow: 'hidden' }}>
                                    <div style={{
                                        background: i < 3 ? 'var(--accent-green)' : 'var(--accent-purple)',
                                        width: `${Math.min(feat.importance / (featureData[0]?.importance || 1) * 100, 100)}%`,
                                        height: '100%',
                                        borderRadius: '4px'
                                    }}></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Year-by-Year Forecasts */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📅 Year-by-Year Forecasts (2025-2030)</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Copper Price Forecast"
                        insight="Both Gradient Boosting and LSTM models predict relatively stable copper prices through 2030, with LSTM showing more conservative estimates."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={forecastChartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" />
                                <YAxis stroke="#71717a" domain={['auto', 'auto']} />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Area type="monotone" dataKey="copper_upper" fill="#22c55e22" stroke="none" name="Upper Bound" />
                                <Area type="monotone" dataKey="copper_lower" fill="#18181b" stroke="none" name="Lower Bound" />
                                <Line type="monotone" dataKey="copper_gb" stroke="#22c55e" strokeWidth={2} name="GB Forecast" dot={{ r: 4 }} />
                                <Line type="monotone" dataKey="copper_lstm" stroke="#a855f7" strokeWidth={2} name="LSTM Forecast" dot={{ r: 4 }} strokeDasharray="5 5" />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Copper Price Predictions</h4>
                        {copperForecasts.map((forecast, i) => (
                            <div key={i} className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <span style={{ fontWeight: 600 }}>{forecast.year}</span>
                                    <span style={{ color: 'var(--accent-green)' }}>
                                        ${forecast.prediction?.toLocaleString()}
                                    </span>
                                </div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                    Range: ${forecast.lower_bound?.toLocaleString()} - ${forecast.upper_bound?.toLocaleString()}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* PCA Analysis */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🎛️ Principal Component Analysis</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Variance Explained"
                        insight={`First principal component explains ${(pcaVariance[0] * 100).toFixed(1)}% of variance. ${pcaData.n_components_95_var || 5} components needed for 95% variance.`}
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={pcaChartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="component" stroke="#71717a" />
                                <YAxis stroke="#71717a" domain={[0, 100]} unit="%" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} formatter={(v) => `${v.toFixed(1)}%`} />
                                <Bar dataKey="variance" fill="#3b82f6" name="Individual" radius={[4, 4, 0, 0]} />
                                <Line type="monotone" dataKey="cumulative" stroke="#22c55e" strokeWidth={2} name="Cumulative" dot={{ r: 4 }} />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>PC1 Loadings (Top Features)</h4>
                        {pcaData.loadings?.PC1 && Object.entries(pcaData.loadings.PC1)
                            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                            .slice(0, 6)
                            .map(([feature, loading], i) => (
                                <div key={i} className="card" style={{ padding: '0.6rem', marginBottom: '0.4rem' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <span style={{ fontSize: '0.85rem' }}>{feature.replace(/_/g, ' ')}</span>
                                        <span style={{
                                            color: loading > 0 ? 'var(--accent-green)' : 'var(--accent-red)',
                                            fontWeight: 600,
                                            fontSize: '0.85rem'
                                        }}>
                                            {loading > 0 ? '+' : ''}{loading.toFixed(3)}
                                        </span>
                                    </div>
                                </div>
                            ))}
                    </div>
                </div>
            </div>

            {/* Market Regimes & Anomalies */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📊 Market Regimes & Anomaly Detection</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Market Regime Distribution"
                        insight={`Current market regime: ${regimeData.current_regime || 'Neutral'}. Based on 12-month rolling analysis of copper prices.`}
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={regimeChartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="regime" stroke="#71717a" />
                                <YAxis stroke="#71717a" />
                                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }} />
                                <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                                    {regimeChartData.map((entry, i) => (
                                        <Cell key={i} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>
                            <span style={{ marginRight: '0.5rem' }}>⚠️</span>
                            Detected Anomalies ({clustering.anomaly_detection?.n_anomalies || 0})
                        </h4>
                        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                            Isolation Forest detected {((clustering.anomaly_detection?.anomaly_rate || 0) * 100).toFixed(1)}% of data as outliers
                        </p>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                            {anomalyDates.slice(0, 10).map((date, i) => (
                                <span key={i} style={{
                                    background: 'rgba(249, 115, 22, 0.2)',
                                    color: 'var(--accent-orange)',
                                    padding: '0.25rem 0.5rem',
                                    borderRadius: '4px',
                                    fontSize: '0.8rem'
                                }}>
                                    {date}
                                </span>
                            ))}
                        </div>
                        <div className="card" style={{ marginTop: '1rem', padding: '1rem', borderLeft: '3px solid var(--accent-orange)' }}>
                            <strong>Notable anomaly periods:</strong>
                            <ul style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.5rem', paddingLeft: '1rem' }}>
                                <li>2000: Dot-com bubble</li>
                                <li>2008-2009: Financial crisis</li>
                                <li>2020: COVID-19 crash</li>
                                <li>2022: Post-pandemic inflation spike</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            {/* Cross-Correlations */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">⏱️ Lead/Lag Relationships</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Cross-Correlation Analysis</h4>
                        {crossCorrData.map((cc, i) => (
                            <div key={i} className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ fontSize: '0.9rem' }}>{cc.feature}</span>
                                    <span style={{
                                        color: cc.lag === 0 ? 'var(--accent-blue)' : cc.lag > 0 ? 'var(--accent-green)' : 'var(--accent-purple)',
                                        fontWeight: 600,
                                        fontSize: '0.85rem'
                                    }}>
                                        {cc.lag === 0 ? 'Concurrent' : `${cc.lag > 0 ? 'Leads' : 'Lags'} by ${Math.abs(cc.lag)}m`}
                                    </span>
                                </div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                    Correlation: {cc.correlation.toFixed(3)}
                                </div>
                            </div>
                        ))}
                    </div>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>Key Lead/Lag Insights</h4>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.75rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <strong>🔮 Predictive Signal:</strong>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                CPI Energy lags Energy Index by 12 months - inflation follows energy prices
                            </p>
                        </div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '0.75rem', borderLeft: '3px solid var(--accent-blue)' }}>
                            <strong>🔗 Co-movement:</strong>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                Gas prices and energy index move together (r=0.93)
                            </p>
                        </div>
                        <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-purple)' }}>
                            <strong>⚡ Metal-Energy Link:</strong>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0' }}>
                                Copper lags energy by 10 months - industrial demand follows energy cycles
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Model Comparison */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🤖 Model Performance Comparison</h3>
                <div style={{ marginTop: '1rem' }}>
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                        Note: Negative R² on test set indicates overfitting or poor time-series handling. For proper forecasting, use LSTM with rolling validation.
                    </p>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                <th style={{ textAlign: 'left', padding: '0.75rem' }}>Model</th>
                                <th style={{ textAlign: 'right', padding: '0.75rem' }}>R² Score</th>
                                <th style={{ textAlign: 'right', padding: '0.75rem' }}>RMSE</th>
                                <th style={{ textAlign: 'right', padding: '0.75rem' }}>MAE</th>
                            </tr>
                        </thead>
                        <tbody>
                            {modelComparisonData.map((model, i) => (
                                <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                                    <td style={{ padding: '0.75rem' }}>
                                        {i === 0 && <span style={{ marginRight: '0.5rem' }}>🥇</span>}
                                        {model.name}
                                    </td>
                                    <td style={{
                                        textAlign: 'right',
                                        padding: '0.75rem',
                                        color: model.r2 > 0 ? 'var(--accent-green)' : 'var(--accent-orange)'
                                    }}>
                                        {model.r2.toFixed(4)}
                                    </td>
                                    <td style={{ textAlign: 'right', padding: '0.75rem' }}>
                                        {model.rmse?.toFixed(2)}
                                    </td>
                                    <td style={{ textAlign: 'right', padding: '0.75rem' }}>
                                        {model.mae?.toFixed(2)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Data Summary */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(168, 85, 247, 0.1))' }}>
                <h3 className="chart-title">📋 Analysis Summary</h3>
                <div style={{ marginTop: '1rem' }}>
                    <div className="grid-2">
                        <div>
                            <h4 style={{ marginBottom: '0.5rem' }}>Dataset Information</h4>
                            <ul style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', paddingLeft: '1.5rem' }}>
                                <li><strong>{metadata.data_points}</strong> monthly observations</li>
                                <li><strong>{metadata.features?.length}</strong> economic indicators</li>
                                <li>Date range: <strong>{metadata.date_range?.start}</strong> to <strong>{metadata.date_range?.end}</strong></li>
                            </ul>
                        </div>
                        <div>
                            <h4 style={{ marginBottom: '0.5rem' }}>Analysis Performed</h4>
                            <ul style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', paddingLeft: '1.5rem' }}>
                                <li>6 ML regression models trained</li>
                                <li>LSTM deep learning forecasting</li>
                                <li>K-Means clustering (4 clusters)</li>
                                <li>PCA dimensionality reduction</li>
                                <li>Isolation Forest anomaly detection</li>
                                <li>Cross-correlation lead/lag analysis</li>
                            </ul>
                        </div>
                    </div>
                    <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(34, 197, 94, 0.2)', borderRadius: '8px', textAlign: 'center' }}>
                        <p style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: '1.1rem', margin: 0 }}>
                            🔬 Comprehensive analysis of commodities, monetary policy, and economic indicators reveals strong interconnections in global markets
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}
