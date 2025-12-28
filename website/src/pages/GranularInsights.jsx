import { useState } from 'react'
import {
    BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell, AreaChart, Area, RadarChart, PolarGrid,
    PolarAngleAxis, Radar, PieChart, Pie
} from 'recharts'

import analysisData from '../data/granular_analysis_complete.json'

const COLORS = ['#6366f1', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899']

export default function GranularInsights() {
    const [activeCategory, setActiveCategory] = useState('energy_grid')
    const categories = analysisData?.categories || {}

    const categoryMeta = {
        energy_grid: { icon: '⚡', label: 'Energy & Grid', color: '#f59e0b' },
        economics: { icon: '💰', label: 'Economics', color: '#22c55e' },
        manufacturing: { icon: '🏭', label: 'Manufacturing', color: '#6366f1' },
        battery_tech: { icon: '🔋', label: 'Battery Tech', color: '#3b82f6' },
        environment: { icon: '🌍', label: 'Environment', color: '#10b981' },
        demographics: { icon: '👥', label: 'Demographics', color: '#8b5cf6' },
        gpu_neural: { icon: '🧠', label: 'AI Models', color: '#ec4899' }
    }

    const totalQuestions = Object.values(categories).reduce(
        (sum, cat) => sum + (cat.questions?.length || 0), 0
    )

    const activeData = categories[activeCategory]?.questions || []

    return (
        <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
            {/* Header */}
            <div style={{
                background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)',
                borderRadius: '20px',
                padding: '2.5rem',
                marginBottom: '2rem',
                color: 'white'
            }}>
                <h1 style={{ fontSize: '2.5rem', fontWeight: '800', marginBottom: '0.5rem' }}>
                    🔬 Granular Deep Dive
                </h1>
                <p style={{ fontSize: '1.1rem', opacity: 0.9, maxWidth: '700px' }}>
                    {totalQuestions} hyper-specific questions answered with GPU-accelerated machine learning.
                    Questions nobody else is asking.
                </p>
                <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
                    <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>{totalQuestions}</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Questions Analyzed</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>7</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Categories</div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1rem 1.5rem', borderRadius: '12px' }}>
                        <div style={{ fontSize: '1.75rem', fontWeight: '700' }}>6</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>ML Models Trained</div>
                    </div>
                </div>
            </div>

            {/* Category Tabs */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                gap: '0.75rem',
                marginBottom: '2rem'
            }}>
                {Object.entries(categoryMeta).map(([key, meta]) => (
                    <button
                        key={key}
                        onClick={() => setActiveCategory(key)}
                        style={{
                            padding: '1rem',
                            borderRadius: '12px',
                            border: 'none',
                            background: activeCategory === key
                                ? `linear-gradient(135deg, ${meta.color}, ${meta.color}dd)`
                                : 'var(--bg-card)',
                            color: activeCategory === key ? 'white' : 'var(--text-secondary)',
                            fontWeight: '600',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            textAlign: 'center'
                        }}
                    >
                        <div style={{ fontSize: '1.5rem', marginBottom: '0.25rem' }}>{meta.icon}</div>
                        <div style={{ fontSize: '0.8rem' }}>{meta.label}</div>
                        <div style={{ fontSize: '0.7rem', opacity: 0.7, marginTop: '0.25rem' }}>
                            {categories[key]?.questions?.length || 0} Q's
                        </div>
                    </button>
                ))}
            </div>

            {/* Questions Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1rem' }}>
                {activeData.map((q, i) => (
                    <QuestionCard key={q.id} question={q} index={i} categoryColor={categoryMeta[activeCategory]?.color} />
                ))}
            </div>
        </div>
    )
}

function QuestionCard({ question, index, categoryColor }) {
    const [expanded, setExpanded] = useState(false)

    return (
        <div
            className="card"
            style={{
                padding: '1.25rem',
                borderLeft: `4px solid ${categoryColor}`,
                cursor: 'pointer',
                transition: 'all 0.2s'
            }}
            onClick={() => setExpanded(!expanded)}
        >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                <span style={{
                    background: categoryColor,
                    color: 'white',
                    padding: '0.25rem 0.5rem',
                    borderRadius: '6px',
                    fontSize: '0.75rem',
                    fontWeight: '600'
                }}>
                    Q{question.id}
                </span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    {expanded ? '▲' : '▼'}
                </span>
            </div>

            <h4 style={{ fontSize: '0.95rem', fontWeight: '600', marginBottom: '0.5rem', lineHeight: 1.3 }}>
                {question.question}
            </h4>

            <div style={{
                fontSize: '1.25rem',
                fontWeight: '700',
                color: categoryColor,
                marginBottom: '0.5rem'
            }}>
                {question.answer}
            </div>

            {question.insight && (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: 0 }}>
                    💡 {question.insight}
                </p>
            )}

            {expanded && question.data && Array.isArray(question.data) && question.data.length > 0 && (
                <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
                    <MiniChart data={question.data} />
                </div>
            )}

            {expanded && question.data && !Array.isArray(question.data) && (
                <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
                    <pre style={{ fontSize: '0.7rem', overflow: 'auto', maxHeight: '200px', background: 'var(--bg-tertiary)', padding: '0.5rem', borderRadius: '6px' }}>
                        {JSON.stringify(question.data, null, 2)}
                    </pre>
                </div>
            )}
        </div>
    )
}

function MiniChart({ data }) {
    if (!data || data.length === 0) return null

    // Detect chart type from data shape
    const keys = Object.keys(data[0])
    const numericKeys = keys.filter(k => typeof data[0][k] === 'number')
    const labelKey = keys.find(k => typeof data[0][k] === 'string') || keys[0]

    if (numericKeys.length === 0) return null

    const dataKey = numericKeys[0]

    return (
        <ResponsiveContainer width="100%" height={150}>
            <BarChart data={data.slice(0, 10)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis
                    dataKey={labelKey}
                    tick={{ fontSize: 10 }}
                    stroke="#71717a"
                    interval={0}
                    angle={-30}
                    textAnchor="end"
                    height={50}
                />
                <YAxis tick={{ fontSize: 10 }} stroke="#71717a" />
                <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #27272a', fontSize: 12 }} />
                <Bar dataKey={dataKey} fill="#6366f1" radius={[4, 4, 0, 0]}>
                    {data.slice(0, 10).map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                </Bar>
            </BarChart>
        </ResponsiveContainer>
    )
}
