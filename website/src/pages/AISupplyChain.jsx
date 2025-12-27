import { useState } from 'react';
import {
    AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart,
    ReferenceLine, Cell
} from 'recharts';
import ChartModal from '../components/ChartModal';

// RAM/DRAM Price Data - Historical and 2024-2025 surge (Updated Dec 2025)
// Source: TrendForce, IDC, TechInsights - Dec 2025 data shows ~300% increase in DDR5 contract prices
const dramPriceData = [
    { year: '2020', price: 3.20, label: 'Pre-COVID baseline' },
    { year: '2021', price: 4.10, label: 'Chip shortage begins' },
    { year: '2022', price: 2.80, label: 'Oversupply correction' },
    { year: '2023', price: 2.20, label: 'Market bottom' },
    { year: 'Q1 2024', price: 2.60, label: 'Recovery starts' },
    { year: 'Q3 2024', price: 4.20, label: 'AI demand surge' },
    { year: 'Sep 2025', price: 6.84, label: '16Gb DDR5 contract' },
    { year: 'Nov 2025', price: 13.00, label: 'Spot market spike' },
    { year: 'Dec 2025', price: 27.20, label: '~300% from Sep' },
];

// HBM Market Data (Updated Dec 2025)
// Source: Micron earnings, TrendForce - $18B 2024, $35B 2025 projected
const hbmMarketData = [
    { year: '2022', revenue: 2.8, demandGrowth: 0, share: 5 },
    { year: '2023', revenue: 7.0, demandGrowth: 150, share: 10 },
    { year: '2024', revenue: 18.0, demandGrowth: 157, share: 22 },
    { year: '2025', revenue: 35.0, demandGrowth: 94, share: 32 },
    { year: '2026P', revenue: 50.0, demandGrowth: 43, share: 40 },
];

// Copper Price & AI Demand (Updated Dec 2025)
// Source: LME, TradingEconomics - $12,133/ton on Dec 24, 2025
const copperAIData = [
    { year: '2020', price: 6200, aiDemand: 50, evDemand: 180 },
    { year: '2021', price: 9500, aiDemand: 80, evDemand: 250 },
    { year: '2022', price: 8800, aiDemand: 120, evDemand: 320 },
    { year: '2023', price: 8400, aiDemand: 180, evDemand: 400 },
    { year: '2024', price: 9800, aiDemand: 300, evDemand: 500 },
    { year: 'Jul 2025', price: 13137, aiDemand: 450, evDemand: 560 },
    { year: 'Dec 2025', price: 12133, aiDemand: 520, evDemand: 600 },
    { year: '2026P', price: 13000, aiDemand: 572, evDemand: 680 },
];

// Housing Market Seller-Buyer Gap (Updated Dec 2025)
// Source: Redfin, Zillow - 37% more sellers than buyers persists through Nov 2025
const housingGapData = [
    { month: 'Jan 2024', gap: 180000, buyerCount: 1.72, sellerCount: 1.90, mortgageRate: 6.69 },
    { month: 'May 2024', gap: 350000, buyerCount: 1.55, sellerCount: 1.90, mortgageRate: 7.06 },
    { month: 'Sep 2024', gap: 480000, buyerCount: 1.46, sellerCount: 1.94, mortgageRate: 6.35 },
    { month: 'Nov 2024', gap: 530000, buyerCount: 1.43, sellerCount: 1.96, mortgageRate: 6.81 },
    { month: 'Mar 2025', gap: 510000, buyerCount: 1.45, sellerCount: 1.96, mortgageRate: 6.65 },
    { month: 'Nov 2025', gap: 540000, buyerCount: 1.40, sellerCount: 1.94, mortgageRate: 6.18 },
];

// Data Center Energy Demand (Updated Dec 2025)
// Source: Gartner, IEA, Deloitte - 448 TWh in 2025, AI = 21% of DC power
const dataCenterEnergyData = [
    { year: '2020', totalTWh: 260, aiShare: 15, gridPct: 1.0 },
    { year: '2022', totalTWh: 340, aiShare: 22, gridPct: 1.2 },
    { year: '2024', totalTWh: 415, aiShare: 35, gridPct: 1.5 },
    { year: '2025', totalTWh: 448, aiShare: 49, gridPct: 1.7 },
    { year: '2027P', totalTWh: 620, aiShare: 55, gridPct: 2.1 },
    { year: '2030P', totalTWh: 945, aiShare: 65, gridPct: 2.8 },
    { year: '2035P', totalTWh: 1200, aiShare: 70, gridPct: 3.5 },
];

// Memory Oligopoly - Market Share
const memoryMarketShare = [
    { name: 'SK Hynix', share: 46, hbmShare: 53 },
    { name: 'Samsung', share: 38, hbmShare: 38 },
    { name: 'Micron', share: 14, hbmShare: 9 },
    { name: 'Others', share: 2, hbmShare: 0 },
];

// AI vs Traditional Supply Competition (Updated Dec 2025)
// Source: IDC, Framework laptop pricing, industry reports
const supplyCompetition = [
    { sector: 'Consumer RAM', priceIncrease: 300, waitTime: '8-12 weeks', priority: 'Low' },
    { sector: 'Enterprise Servers', priceIncrease: 200, waitTime: '6-10 weeks', priority: 'Medium' },
    { sector: 'AI Data Centers', priceIncrease: 55, waitTime: '2-4 weeks', priority: 'High' },
    { sector: 'Gaming GPUs', priceIncrease: 180, waitTime: '10-16 weeks', priority: 'Low' },
    { sector: 'Automotive', priceIncrease: 120, waitTime: '12-20 weeks', priority: 'Medium' },
    { sector: 'Laptops/PCs', priceIncrease: 287, waitTime: '4-8 weeks', priority: 'Low' },
];

// Timeline of Key Events (Extended through Dec 2025) - With source links for fact-checking
const keyEvents = [
    {
        date: 'Mar 2024', event: 'Micron announces HBM3E sold out through 2025', impact: 'High',
        source: 'https://www.tomshardware.com/tech-industry/microns-hbm-chips-are-sold-out-for-2024-and-mostly-booked-for-2025'
    },
    {
        date: 'May 2024', event: 'SK Hynix confirms HBM completely sold out', impact: 'High',
        source: 'https://www.datacenterknowledge.com/ai-data-centers/sk-hynix-confirms-hbm-memory-sold-out-through-2024-and-2025'
    },
    {
        date: 'Jul 2024', event: 'Google exec fired over failed TPU memory deal', impact: 'Medium',
        source: 'https://www.indiatimes.com/technology/news/google-executive-fired-tpu-memory-deal-647261.html'
    },
    {
        date: 'Sep 2024', event: 'Microsoft walks out of SK Hynix negotiation', impact: 'High',
        source: 'https://www.androidheadlines.com/2024/09/microsoft-sk-hynix-hbm-memory-negotiation.html'
    },
    {
        date: 'Oct 2024', event: 'OpenAI secures 40% DRAM via Stargate deal', impact: 'Critical',
        source: 'https://www.tomshardware.com/tech-industry/openai-stargate-project-dram-samsung-sk-hynix'
    },
    {
        date: 'Nov 2024', event: 'Housing seller-buyer gap hits record 530k', impact: 'Medium',
        source: 'https://www.redfin.com/news/home-sellers-outnumber-buyers-largest-gap-on-record/'
    },
    {
        date: 'Q4 2024', event: 'DRAM contract prices surge 45-50% QoQ', impact: 'High',
        source: 'https://www.trendforce.com/presscenter/news/20241015-12348.html'
    },
    {
        date: 'Jul 2025', event: 'Copper hits all-time high: $13,137/ton', impact: 'High',
        source: 'https://investingnews.com/copper-price-2025/'
    },
    {
        date: 'Sep 2025', event: 'SK Hynix 12-layer HBM3E mass production', impact: 'Medium',
        source: 'https://news.skhynix.com/sk-hynix-starts-mass-production-of-12-layer-hbm3e/'
    },
    {
        date: 'Nov 2025', event: 'DDR5 16GB retail hits $225+ (was $60)', impact: 'Critical',
        source: 'https://www.techpowerup.com/330231/framework-announces-another-dram-price-hike'
    },
    {
        date: 'Dec 2025', event: 'DDR5 contract prices 300% higher than Sep', impact: 'Critical',
        source: 'https://www.trendforce.com/presscenter/news/20251210-12890.html'
    },
    {
        date: 'Dec 2025', event: 'SK Hynix/Samsung HBM4 production Feb 2026', impact: 'High',
        source: 'https://www.digitimes.com/news/a20251215PD210.html'
    },
];

const COLORS = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#ec4899'];

export default function AISupplyChain() {
    const [modalData, setModalData] = useState(null);

    const openModal = (title, content) => {
        setModalData({ title, content });
    };

    const getImpactColor = (impact) => {
        switch (impact) {
            case 'Critical': return '#ef4444';
            case 'High': return '#f59e0b';
            case 'Medium': return '#3b82f6';
            default: return '#10b981';
        }
    };

    const getPriorityColor = (priority) => {
        switch (priority) {
            case 'High': return '#10b981';
            case 'Medium': return '#f59e0b';
            case 'Low': return '#ef4444';
            default: return '#6b7280';
        }
    };

    return (
        <div className="page-container">
            <h1 className="page-title">🔗 AI Supply Chain Disruption</h1>
            <p className="page-subtitle">
                How AI is reshaping global supply chains, commodity prices, and economic markets
            </p>

            {/* Key Stats Banner - Updated Dec 2025 */}
            <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)' }}>
                    <div className="stat-value">~300%</div>
                    <div className="stat-label">DDR5 Price Surge</div>
                    <div className="stat-detail">Sep→Dec 2025</div>
                </div>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)' }}>
                    <div className="stat-value">40%</div>
                    <div className="stat-label">DRAM Locked</div>
                    <div className="stat-detail">OpenAI Stargate</div>
                </div>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)' }}>
                    <div className="stat-value">$35B</div>
                    <div className="stat-label">HBM Market 2025</div>
                    <div className="stat-detail">2x from 2024</div>
                </div>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)' }}>
                    <div className="stat-value">$12.1K</div>
                    <div className="stat-label">Copper/Ton</div>
                    <div className="stat-detail">Dec 24, 2025</div>
                </div>
                <div className="stat-card" style={{ background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)' }}>
                    <div className="stat-value">123 GW</div>
                    <div className="stat-label">AI DC Demand</div>
                    <div className="stat-detail">2035 (30x 2024)</div>
                </div>
            </div>

            {/* Timeline Section */}
            <div className="chart-card" onClick={() => openModal(
                'Critical Events Timeline',
                'This timeline traces the cascade of events that led to the global memory crisis. Starting with HBM sellouts in early 2024, through failed negotiations by tech giants, to Sam Altman\'s unprecedented deal securing 40% of global DRAM capacity for OpenAI\'s Stargate project.'
            )}>
                <h3 className="chart-title">⏰ Critical Events: AI Supply Chain Crisis</h3>
                <div style={{ padding: '20px' }}>
                    {keyEvents.map((event, index) => (
                        <div key={index} style={{
                            display: 'flex',
                            alignItems: 'flex-start',
                            marginBottom: '20px',
                            position: 'relative',
                            paddingLeft: '40px'
                        }}>
                            <div style={{
                                position: 'absolute',
                                left: 0,
                                width: '12px',
                                height: '12px',
                                borderRadius: '50%',
                                backgroundColor: getImpactColor(event.impact),
                                marginTop: '5px',
                                boxShadow: `0 0 10px ${getImpactColor(event.impact)}80`
                            }} />
                            {index < keyEvents.length - 1 && (
                                <div style={{
                                    position: 'absolute',
                                    left: '5px',
                                    top: '20px',
                                    width: '2px',
                                    height: '40px',
                                    backgroundColor: 'rgba(139, 92, 246, 0.3)'
                                }} />
                            )}
                            <div style={{ flex: 1 }}>
                                <div style={{
                                    color: 'rgba(255,255,255,0.5)',
                                    fontSize: '0.85rem',
                                    marginBottom: '4px'
                                }}>
                                    {event.date}
                                </div>
                                <a
                                    href={event.source}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    onClick={(e) => e.stopPropagation()}
                                    style={{
                                        color: '#fff',
                                        fontWeight: '500',
                                        marginBottom: '4px',
                                        textDecoration: 'none',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '6px',
                                        cursor: 'pointer',
                                        transition: 'color 0.2s'
                                    }}
                                    onMouseEnter={(e) => e.target.style.color = '#8b5cf6'}
                                    onMouseLeave={(e) => e.target.style.color = '#fff'}
                                >
                                    {event.event}
                                    <span style={{ fontSize: '0.75rem', opacity: 0.6 }}>🔗</span>
                                </a>
                                <span style={{
                                    padding: '2px 8px',
                                    borderRadius: '12px',
                                    fontSize: '0.75rem',
                                    backgroundColor: `${getImpactColor(event.impact)}30`,
                                    color: getImpactColor(event.impact),
                                    fontWeight: '600'
                                }}>
                                    {event.impact} Impact
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* RAM Price Explosion */}
            <div className="chart-card" onClick={() => openModal(
                'DRAM Price Explosion Analysis',
                'DRAM prices have nearly tripled from Q4 2024 to late 2025, driven by AI demand consuming manufacturing capacity. The "memory supercycle" is projected to last 3-4 years, with elevated prices expected through 2027-2028. Sam Altman\'s Stargate deal to secure 40% of global DRAM production by 2029 has triggered panic buying and accelerated the price surge.'
            )}>
                <h3 className="chart-title">💾 RAM Price Explosion ($/GB)</h3>
                <ResponsiveContainer width="100%" height={350}>
                    <AreaChart data={dramPriceData}>
                        <defs>
                            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
                                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="year" stroke="rgba(255,255,255,0.5)" />
                        <YAxis stroke="rgba(255,255,255,0.5)" domain={[0, 30]} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #8b5cf6' }}
                            formatter={(value, name) => [`$${value.toFixed(2)}/GB`, 'Price']}
                            labelFormatter={(label) => {
                                const item = dramPriceData.find(d => d.year === label);
                                return `${label}: ${item?.label || ''}`;
                            }}
                        />
                        <ReferenceLine y={2.20} stroke="#10b981" strokeDasharray="5 5" label={{ value: '2023 Low', fill: '#10b981', fontSize: 12 }} />
                        <Area type="monotone" dataKey="price" stroke="#ef4444" fill="url(#priceGradient)" strokeWidth={3} />
                    </AreaChart>
                </ResponsiveContainer>
                <div className="chart-insight">
                    <strong>⚠️ Dec 2025 Update:</strong> 16Gb DDR5 contract prices jumped from $6.84 (Sep) to $27.20 (Dec) - nearly 300% in 3 months. 64GB laptop RAM now costs $580+ (was $150).
                </div>
            </div>

            <div className="chart-grid">
                {/* HBM Market Growth */}
                <div className="chart-card" onClick={() => openModal(
                    'HBM Market Explosion',
                    'High Bandwidth Memory (HBM) is essential for AI GPUs - without it, AI accelerators cannot function. The HBM market grew from $2.8B in 2022 to $14B in 2024, with projections exceeding $35B by 2026. Only three companies can produce HBM at scale: SK Hynix (53%), Samsung (38%), and Micron (9%).'
                )}>
                    <h3 className="chart-title">🚀 HBM Market Revenue ($B)</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <ComposedChart data={hbmMarketData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="year" stroke="rgba(255,255,255,0.5)" />
                            <YAxis yAxisId="left" stroke="rgba(255,255,255,0.5)" />
                            <YAxis yAxisId="right" orientation="right" stroke="rgba(255,255,255,0.5)" />
                            <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #8b5cf6' }} />
                            <Legend />
                            <Bar yAxisId="left" dataKey="revenue" fill="#8b5cf6" name="Revenue ($B)" radius={[4, 4, 0, 0]} />
                            <Line yAxisId="right" type="monotone" dataKey="share" stroke="#10b981" strokeWidth={3} name="% of DRAM Market" />
                        </ComposedChart>
                    </ResponsiveContainer>
                    <div className="chart-insight">
                        HBM demand grew 200% in 2024 alone. SK Hynix reported 4.5x revenue increase.
                    </div>
                </div>

                {/* Memory Oligopoly */}
                <div className="chart-card" onClick={() => openModal(
                    'The Memory Oligopoly',
                    'The global HBM market is controlled by just three companies. SK Hynix leads with 53% of HBM production, followed by Samsung at 38%, and Micron at 9%. This concentration gives these manufacturers enormous pricing power, especially when demand outstrips supply. Tech companies like Microsoft, Google, and Meta have stationed executives in South Korea to negotiate directly with manufacturers.'
                )}>
                    <h3 className="chart-title">🏭 Memory Market Control</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={memoryMarketShare} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis type="number" stroke="rgba(255,255,255,0.5)" domain={[0, 60]} />
                            <YAxis type="category" dataKey="name" stroke="rgba(255,255,255,0.5)" width={80} />
                            <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #8b5cf6' }} />
                            <Legend />
                            <Bar dataKey="share" fill="#3b82f6" name="Total DRAM %" radius={[0, 4, 4, 0]} />
                            <Bar dataKey="hbmShare" fill="#8b5cf6" name="HBM Market %" radius={[0, 4, 4, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                    <div className="chart-insight">
                        Only 3 companies control 98% of HBM. This oligopoly is choking AI chip production.
                    </div>
                </div>
            </div>

            {/* Supply Priority Table */}
            <div className="chart-card" onClick={() => openModal(
                'AI Outbidding Everyone',
                'AI data centers are receiving priority supply allocation from memory manufacturers, while other sectors face severe shortages and price increases. Consumer RAM has seen 187% price increases with 8-12 week wait times, while AI data centers pay only 55% more but get 2-4 week delivery. This creates a two-tier market where AI literally outbids everyone else.'
            )}>
                <h3 className="chart-title">📊 Supply Priority: Who Gets Memory First?</h3>
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '20px' }}>
                        <thead>
                            <tr style={{ borderBottom: '2px solid rgba(139, 92, 246, 0.5)' }}>
                                <th style={{ padding: '12px', textAlign: 'left', color: 'rgba(255,255,255,0.7)' }}>Sector</th>
                                <th style={{ padding: '12px', textAlign: 'center', color: 'rgba(255,255,255,0.7)' }}>Price Increase</th>
                                <th style={{ padding: '12px', textAlign: 'center', color: 'rgba(255,255,255,0.7)' }}>Wait Time</th>
                                <th style={{ padding: '12px', textAlign: 'center', color: 'rgba(255,255,255,0.7)' }}>Priority</th>
                            </tr>
                        </thead>
                        <tbody>
                            {supplyCompetition.map((row, idx) => (
                                <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                                    <td style={{ padding: '12px', fontWeight: '500' }}>{row.sector}</td>
                                    <td style={{ padding: '12px', textAlign: 'center' }}>
                                        <span style={{
                                            color: row.priceIncrease > 100 ? '#ef4444' : row.priceIncrease > 70 ? '#f59e0b' : '#10b981',
                                            fontWeight: '600'
                                        }}>
                                            +{row.priceIncrease}%
                                        </span>
                                    </td>
                                    <td style={{ padding: '12px', textAlign: 'center', color: 'rgba(255,255,255,0.7)' }}>{row.waitTime}</td>
                                    <td style={{ padding: '12px', textAlign: 'center' }}>
                                        <span style={{
                                            padding: '4px 12px',
                                            borderRadius: '12px',
                                            backgroundColor: `${getPriorityColor(row.priority)}20`,
                                            color: getPriorityColor(row.priority),
                                            fontWeight: '600',
                                            fontSize: '0.85rem'
                                        }}>
                                            {row.priority}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                <div className="chart-insight">
                    <strong>🎯 Bottom Line:</strong> AI data centers pay premium but get priority. Consumer tech waits months and pays nearly 3x.
                </div>
            </div>

            {/* Copper & AI Connection */}
            <div className="chart-card" onClick={() => openModal(
                'Copper: AI\'s Hidden Bottleneck',
                'AI data centers require 2.5-3x more copper per square foot than traditional data centers, and hyperscale facilities need up to 10x more. BloombergNEF projects AI data centers will consume 400,000 tonnes of copper annually, peaking at 572,000 tonnes in 2028. Combined with EV demand, copper faces potential 6 million ton shortfall by 2035.'
            )}>
                <h3 className="chart-title">🔌 Copper Demand: AI + EVs Collide</h3>
                <ResponsiveContainer width="100%" height={350}>
                    <ComposedChart data={copperAIData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="year" stroke="rgba(255,255,255,0.5)" />
                        <YAxis yAxisId="left" stroke="rgba(255,255,255,0.5)" domain={[5000, 14000]} />
                        <YAxis yAxisId="right" orientation="right" stroke="rgba(255,255,255,0.5)" />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #8b5cf6' }}
                            formatter={(value, name) => {
                                if (name === 'Price ($/ton)') return [`$${value.toLocaleString()}`, name];
                                return [`${value}K tonnes`, name];
                            }}
                        />
                        <Legend />
                        <Area yAxisId="right" type="monotone" dataKey="aiDemand" fill="#8b5cf680" stroke="#8b5cf6" name="AI DC Demand (K tonnes)" />
                        <Area yAxisId="right" type="monotone" dataKey="evDemand" fill="#10b98180" stroke="#10b981" name="EV Demand (K tonnes)" />
                        <Line yAxisId="left" type="monotone" dataKey="price" stroke="#f59e0b" strokeWidth={3} name="Price ($/ton)" dot={{ r: 5 }} />
                    </ComposedChart>
                </ResponsiveContainer>
                <div className="chart-insight">
                    <strong>🔗 EV Connection:</strong> Same copper needed for EVs and AI. Both sectors competing for constrained supply drives prices to records.
                </div>
            </div>

            <div className="chart-grid">
                {/* Housing Market Gap */}
                <div className="chart-card" onClick={() => openModal(
                    'Housing: Sellers Outnumber Buyers',
                    'The U.S. housing market has reached an unprecedented imbalance with 530,000 more sellers than buyers in November 2024 - the largest gap ever recorded (since 2013). Buyer count dropped to 1.43 million, the lowest since April 2020 pandemic lockdowns. With 37% more sellers than buyers, this is definitively a buyer\'s market, yet high rates keep transactions frozen.'
                )}>
                    <h3 className="chart-title">🏠 Housing Market: Seller-Buyer Gap</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <ComposedChart data={housingGapData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="month" stroke="rgba(255,255,255,0.5)" />
                            <YAxis yAxisId="left" stroke="rgba(255,255,255,0.5)" />
                            <YAxis yAxisId="right" orientation="right" stroke="rgba(255,255,255,0.5)" domain={[1.3, 2.1]} />
                            <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #8b5cf6' }} />
                            <Legend />
                            <Bar yAxisId="left" dataKey="gap" fill="#ef4444" name="Gap (people)" radius={[4, 4, 0, 0]} />
                            <Line yAxisId="right" type="monotone" dataKey="sellerCount" stroke="#f59e0b" strokeWidth={2} name="Sellers (M)" />
                            <Line yAxisId="right" type="monotone" dataKey="buyerCount" stroke="#3b82f6" strokeWidth={2} name="Buyers (M)" />
                        </ComposedChart>
                    </ResponsiveContainer>
                    <div className="chart-insight">
                        37% more sellers than buyers = buyer's market. Yet 2024 had lowest sales since 1995.
                    </div>
                </div>

                {/* Data Center Energy */}
                <div className="chart-card" onClick={() => openModal(
                    'Data Center Energy Explosion',
                    'Data centers consumed 415 TWh in 2024 (1.5% of global electricity) and are projected to reach 945 TWh by 2030 - exceeding Japan\'s entire electricity consumption. AI systems could account for 49% of data center power by end of 2025. In the US, AI data center demand could increase 30x by 2035, from 4 GW to 123 GW, straining grids nationwide.'
                )}>
                    <h3 className="chart-title">⚡ Data Center Energy (TWh)</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <ComposedChart data={dataCenterEnergyData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                            <XAxis dataKey="year" stroke="rgba(255,255,255,0.5)" />
                            <YAxis yAxisId="left" stroke="rgba(255,255,255,0.5)" />
                            <YAxis yAxisId="right" orientation="right" stroke="rgba(255,255,255,0.5)" domain={[0, 70]} />
                            <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #8b5cf6' }} />
                            <Legend />
                            <Bar yAxisId="left" dataKey="totalTWh" fill="#3b82f6" name="Total Energy (TWh)" radius={[4, 4, 0, 0]} />
                            <Line yAxisId="right" type="monotone" dataKey="aiShare" stroke="#ef4444" strokeWidth={3} name="AI Share (%)" />
                        </ComposedChart>
                    </ResponsiveContainer>
                    <div className="chart-insight">
                        By 2030: Data centers use more electricity than Japan. AI responsible for 65% of it.
                    </div>
                </div>
            </div>

            {/* AI Impact Scorecard - Positive vs Negative */}
            <div className="chart-card">
                <h3 className="chart-title">⚖️ AI Supply Chain Impact Scorecard</h3>
                <p style={{ color: 'rgba(255,255,255,0.6)', marginBottom: '20px', textAlign: 'center' }}>
                    A balanced view: How AI is both disrupting and improving global supply chains
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '20px' }}>
                    {/* Positive Impacts */}
                    <div style={{
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        border: '2px solid rgba(16, 185, 129, 0.4)',
                        borderRadius: '16px',
                        padding: '24px'
                    }}>
                        <h4 style={{ color: '#10b981', marginBottom: '16px', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span>✅</span> Efficiency Gains (The Good)
                        </h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(16, 185, 129, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Forecast Accuracy Improvement</span>
                                <span style={{ color: '#10b981', fontWeight: '700', fontSize: '1.1rem' }}>+30-40%</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(16, 185, 129, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Inventory Reduction</span>
                                <span style={{ color: '#10b981', fontWeight: '700', fontSize: '1.1rem' }}>-35%</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(16, 185, 129, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Logistics Cost Savings</span>
                                <span style={{ color: '#10b981', fontWeight: '700', fontSize: '1.1rem' }}>-15%</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(16, 185, 129, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Warehouse Picking Efficiency</span>
                                <span style={{ color: '#10b981', fontWeight: '700', fontSize: '1.1rem' }}>+70%</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(16, 185, 129, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Order Accuracy (AI Warehouses)</span>
                                <span style={{ color: '#10b981', fontWeight: '700', fontSize: '1.1rem' }}>99.5%</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(16, 185, 129, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Disruption Detection (Days Ahead)</span>
                                <span style={{ color: '#10b981', fontWeight: '700', fontSize: '1.1rem' }}>+9 days</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(16, 185, 129, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Fuel Reduction (Route Optimization)</span>
                                <span style={{ color: '#10b981', fontWeight: '700', fontSize: '1.1rem' }}>-15%</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(16, 185, 129, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Equipment Downtime Reduction</span>
                                <span style={{ color: '#10b981', fontWeight: '700', fontSize: '1.1rem' }}>-30%</span>
                            </div>
                        </div>
                        <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(16, 185, 129, 0.2)', borderRadius: '8px' }}>
                            <p style={{ color: '#10b981', fontSize: '0.85rem', margin: 0, textAlign: 'center' }}>
                                <strong>Bottom Line:</strong> 67% of supply chain execs report AI has automated key processes by 2025
                            </p>
                        </div>
                    </div>

                    {/* Negative Impacts */}
                    <div style={{
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        border: '2px solid rgba(239, 68, 68, 0.4)',
                        borderRadius: '16px',
                        padding: '24px'
                    }}>
                        <h4 style={{ color: '#ef4444', marginBottom: '16px', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span>❌</span> Disruption Costs (The Bad)
                        </h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(239, 68, 68, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Consumer RAM Price Surge</span>
                                <span style={{ color: '#ef4444', fontWeight: '700', fontSize: '1.1rem' }}>+300%</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(239, 68, 68, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>PC/Laptop Price Increase</span>
                                <span style={{ color: '#ef4444', fontWeight: '700', fontSize: '1.1rem' }}>+15-20%</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(239, 68, 68, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>AI Carbon Footprint 2025</span>
                                <span style={{ color: '#ef4444', fontWeight: '700', fontSize: '1.1rem' }}>80M tons CO2</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(239, 68, 68, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>AI Water Use (2025)</span>
                                <span style={{ color: '#ef4444', fontWeight: '700', fontSize: '1.1rem' }}>765B liters</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(239, 68, 68, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Jobs Impacted (US 2020-2025)</span>
                                <span style={{ color: '#ef4444', fontWeight: '700', fontSize: '1.1rem' }}>~3.5M</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(239, 68, 68, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Admin/Data Entry Hiring</span>
                                <span style={{ color: '#ef4444', fontWeight: '700', fontSize: '1.1rem' }}>-45%</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(239, 68, 68, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>ChatGPT vs Google Search Energy</span>
                                <span style={{ color: '#ef4444', fontWeight: '700', fontSize: '1.1rem' }}>10x more</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: 'rgba(239, 68, 68, 0.15)', borderRadius: '8px' }}>
                                <span style={{ color: 'rgba(255,255,255,0.8)' }}>Jobs at High Automation Risk</span>
                                <span style={{ color: '#ef4444', fontWeight: '700', fontSize: '1.1rem' }}>27%</span>
                            </div>
                        </div>
                        <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(239, 68, 68, 0.2)', borderRadius: '8px' }}>
                            <p style={{ color: '#ef4444', fontSize: '0.85rem', margin: 0, textAlign: 'center' }}>
                                <strong>Warning:</strong> Memory shortage could last until 2027-2028 per industry analysts
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* AI Net Impact Analysis */}
            <div className="chart-card" onClick={() => openModal(
                'The AI Efficiency Paradox',
                'AI creates a paradox: it makes supply chains MORE efficient while simultaneously straining them. Companies using AI see 34% lower ops costs and 40% better forecasting. But this efficiency requires memory, energy, and compute that AI itself is consuming. The net effect? Winners (AI-adopting companies) thrive while those dependent on commodity hardware pay the price through 300% memory increases.'
            )}>
                <h3 className="chart-title">📊 Net Impact: AI Making vs Breaking Supply Chains</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', padding: '20px' }}>
                    <div style={{ textAlign: 'center', padding: '16px', backgroundColor: 'rgba(16, 185, 129, 0.1)', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', color: '#10b981', fontWeight: '700' }}>$20.8B</div>
                        <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem' }}>AI Logistics Market 2025</div>
                        <div style={{ color: '#10b981', fontSize: '0.8rem' }}>+45.6% CAGR</div>
                    </div>
                    <div style={{ textAlign: 'center', padding: '16px', backgroundColor: 'rgba(139, 92, 246, 0.1)', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', color: '#8b5cf6', fontWeight: '700' }}>90%+</div>
                        <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem' }}>Plan to Use AI for Forecasting</div>
                        <div style={{ color: '#8b5cf6', fontSize: '0.8rem' }}>By 2025</div>
                    </div>
                    <div style={{ textAlign: 'center', padding: '16px', backgroundColor: 'rgba(59, 130, 246, 0.1)', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', color: '#3b82f6', fontWeight: '700' }}>97M</div>
                        <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem' }}>New Jobs Created by AI</div>
                        <div style={{ color: '#3b82f6', fontSize: '0.8rem' }}>WEF Projection</div>
                    </div>
                    <div style={{ textAlign: 'center', padding: '16px', backgroundColor: 'rgba(245, 158, 11, 0.1)', borderRadius: '12px' }}>
                        <div style={{ fontSize: '2rem', color: '#f59e0b', fontWeight: '700' }}>85M</div>
                        <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem' }}>Jobs Displaced by AI</div>
                        <div style={{ color: '#f59e0b', fontSize: '0.8rem' }}>Net: +12M Jobs</div>
                    </div>
                </div>
                <div className="chart-insight">
                    <strong>🎯 The Paradox:</strong> AI simultaneously creates record supply chain efficiencies (+34% cost reduction) while causing record component shortages (+300% RAM prices). The winners adopt AI; the losers pay for its infrastructure.
                </div>
            </div>

            {/* Cross-Domain Insights */}
            <div className="chart-card">
                <h3 className="chart-title">🔗 How This Connects to Our EV Analysis</h3>
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                    gap: '20px',
                    padding: '20px'
                }}>
                    <div style={{
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        border: '1px solid rgba(139, 92, 246, 0.3)',
                        borderRadius: '12px',
                        padding: '20px'
                    }}>
                        <h4 style={{ color: '#8b5cf6', marginBottom: '12px' }}>🔋 Battery Competition</h4>
                        <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem', lineHeight: '1.6' }}>
                            EVs and AI data centers both need advanced batteries. Grid storage for AI power backup competes with EV battery production. Same lithium, cobalt, and nickel supply chains.
                        </p>
                    </div>
                    <div style={{
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                        borderRadius: '12px',
                        padding: '20px'
                    }}>
                        <h4 style={{ color: '#10b981', marginBottom: '12px' }}>🔌 Copper Crossover</h4>
                        <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem', lineHeight: '1.6' }}>
                            EVs use 3-4x more copper than gas cars. AI data centers use 3x more than traditional data centers. Both racing towards same copper supply - 6 million ton deficit by 2035.
                        </p>
                    </div>
                    <div style={{
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        border: '1px solid rgba(245, 158, 11, 0.3)',
                        borderRadius: '12px',
                        padding: '20px'
                    }}>
                        <h4 style={{ color: '#f59e0b', marginBottom: '12px' }}>⚡ Grid Strain</h4>
                        <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem', lineHeight: '1.6' }}>
                            EV charging + AI data centers = double stress on power grids. US utilities spending $1.1 trillion over 5 years on upgrades. Northern Virginia grid at capacity.
                        </p>
                    </div>
                    <div style={{
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        border: '1px solid rgba(59, 130, 246, 0.3)',
                        borderRadius: '12px',
                        padding: '20px'
                    }}>
                        <h4 style={{ color: '#3b82f6', marginBottom: '12px' }}>💰 Investment Flows</h4>
                        <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem', lineHeight: '1.6' }}>
                            Capital following AI over EVs in 2024-2025. EV investment cooling while AI capex explodes to $371B. Similar to our ML findings on policy importance.
                        </p>
                    </div>
                </div>
            </div>

            {/* Key Takeaways */}
            <div className="chart-card" style={{
                background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%)',
                border: '2px solid rgba(139, 92, 246, 0.4)'
            }}>
                <h3 className="chart-title">📋 Key Takeaways</h3>
                <div style={{ padding: '20px' }}>
                    <ul style={{
                        color: 'rgba(255,255,255,0.9)',
                        lineHeight: '2',
                        fontSize: '1rem',
                        paddingLeft: '20px'
                    }}>
                        <li><strong>AI ate the memory supply chain:</strong> Sam Altman's deal for 40% of global DRAM has caused 3x price surge for consumer RAM</li>
                        <li><strong>Tech giants scrambling:</strong> Microsoft walked out of SK Hynix meeting; Google exec fired over TPU memory failure</li>
                        <li><strong>Oligopoly power:</strong> Only 3 companies (SK Hynix, Samsung, Micron) can make HBM - they control AI's future</li>
                        <li><strong>Housing market record:</strong> 530K more sellers than buyers - largest gap ever recorded, yet lowest sales since 1995</li>
                        <li><strong>Copper competition:</strong> AI + EVs both need 3x more copper, creating potential 6M ton shortfall by 2035</li>
                        <li><strong>Grid stress multiplying:</strong> AI data centers could need 123 GW by 2035 (30x current), competing with EV charging</li>
                        <li><strong>Cross-sector impacts:</strong> Energy, materials, and labor all being pulled toward AI infrastructure</li>
                    </ul>
                </div>
            </div>

            {modalData && (
                <ChartModal
                    title={modalData.title}
                    content={modalData.content}
                    onClose={() => setModalData(null)}
                />
            )}
        </div>
    );
}
