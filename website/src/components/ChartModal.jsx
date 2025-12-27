import { useState } from 'react'

export default function ChartModal({ title, insight, children, height = 200 }) {
    const [expanded, setExpanded] = useState(false)

    return (
        <>
            {/* Compact preview */}
            <div
                className="chart-container"
                style={{ cursor: 'pointer' }}
                onClick={() => setExpanded(true)}
            >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 className="chart-title">{title}</h3>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Click to expand ↗</span>
                </div>
                <div style={{ height, overflow: 'hidden' }}>
                    {children}
                </div>
            </div>

            {/* Expanded modal */}
            {expanded && (
                <div
                    className="modal-overlay"
                    onClick={() => setExpanded(false)}
                    style={{
                        position: 'fixed',
                        inset: 0,
                        background: 'rgba(0,0,0,0.85)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        zIndex: 1000,
                        padding: '2rem',
                    }}
                >
                    <div
                        className="modal-content"
                        onClick={e => e.stopPropagation()}
                        style={{
                            background: 'var(--bg-card)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '16px',
                            padding: '2rem',
                            maxWidth: '900px',
                            width: '100%',
                            maxHeight: '90vh',
                            overflow: 'auto',
                        }}
                    >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <h2 style={{ fontSize: '1.5rem' }}>{title}</h2>
                            <button
                                onClick={() => setExpanded(false)}
                                style={{
                                    background: 'var(--bg-hover)',
                                    border: 'none',
                                    color: 'var(--text-primary)',
                                    padding: '0.5rem 1rem',
                                    borderRadius: '8px',
                                    cursor: 'pointer',
                                    fontSize: '0.9rem',
                                }}
                            >
                                ✕ Close
                            </button>
                        </div>

                        <div style={{ height: 400 }}>
                            {children}
                        </div>

                        {insight && (
                            <div style={{
                                marginTop: '1.5rem',
                                padding: '1rem',
                                background: 'var(--bg-hover)',
                                borderRadius: '8px',
                                borderLeft: '3px solid var(--accent-green)'
                            }}>
                                <h4 style={{ marginBottom: '0.5rem', color: 'var(--accent-green)' }}>💡 Key Insight</h4>
                                <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6 }}>{insight}</p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </>
    )
}
