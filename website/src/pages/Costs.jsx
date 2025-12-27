import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, Legend, ComposedChart, Area } from 'recharts'
import ChartModal from '../components/ChartModal'
import costData from '../data/cost_analysis.json'

export default function Costs() {
    const electricity = costData.electricity_by_state
    const tou = costData.time_of_use
    const depreciation = costData.depreciation
    const insurance = costData.insurance
    const repair = costData.repair_costs
    const leaseBuy = costData.lease_vs_buy
    const tco = costData.tco
    const charts = costData.charts
    const insights = costData.key_insights

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">💰 Comprehensive Cost Analysis</h1>
                <p className="page-subtitle">Real data on electricity costs, depreciation, insurance, repairs, and total cost of ownership</p>
            </header>

            {/* Key Insights */}
            <div className="stats-grid">
                {insights.slice(0, 4).map((insight, i) => (
                    <div key={i} className="stat-card">
                        <div className="stat-icon">{insight.icon}</div>
                        <div className="stat-value" style={{ fontSize: '0.95rem' }}>{insight.title}</div>
                        <div className="stat-label" style={{ fontSize: '0.7rem' }}>{insight.detail}</div>
                    </div>
                ))}
            </div>

            {/* Electricity Costs by State */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">⚡ Electricity Costs by State - EV Annual Fuel Cost</h3>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.85rem' }}>
                    Based on EIA 2024 residential electricity rates. Assumes 15,000 miles/year, 3.3 mi/kWh efficiency.
                </p>

                <div className="grid-2">
                    <ChartModal
                        title="EV vs Gas Annual Fuel Cost by State"
                        insight="EVs are cheaper to fuel in every US state - even Hawaii with 32¢/kWh electricity. In Louisiana at 9.4¢/kWh, EV owners pay just $420/year vs $1,800 for gas - 77% savings!"
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={charts.electricity_comparison}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="state" stroke="#71717a" angle={-20} textAnchor="end" height={60} />
                                <YAxis stroke="#71717a" tickFormatter={(v) => `$${v}`} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => `$${v.toLocaleString()}`}
                                />
                                <Legend />
                                <Bar dataKey="ev_annual" name="EV (home charging)" fill="#22c55e" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="gas_annual" name="Gas (30 MPG)" fill="#ef4444" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>💵 Cheapest States for EV Charging</h4>
                        <div style={{ maxHeight: '280px', overflowY: 'auto' }}>
                            {electricity.cheapest_states.slice(0, 6).map((state, i) => (
                                <div key={i} className="card" style={{ padding: '0.6rem', marginBottom: '0.4rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <div>
                                        <span style={{ fontWeight: 600 }}>{i + 1}. {state.state}</span>
                                        <span style={{ marginLeft: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{state.cents_kwh}¢/kWh</span>
                                    </div>
                                    <div style={{ textAlign: 'right' }}>
                                        <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>${state.annual_fuel_cost}/yr</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                        <div className="card" style={{ padding: '0.75rem', marginTop: '0.75rem', background: 'rgba(239, 68, 68, 0.1)', borderLeft: '3px solid #ef4444' }}>
                            <strong>⚠️ Most Expensive:</strong> Hawaii @ ${electricity.most_expensive_states[0].annual_fuel_cost}/yr
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                                Still saves ${(3200 - 1474).toLocaleString()}/yr vs gas!
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Time-of-Use Charging */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">⏰ Time-of-Use Charging Savings</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>PG&E California Example</h4>
                        <div className="card" style={{ padding: '1rem', marginBottom: '1rem' }}>
                            {Object.entries(tou.example_california_pge.rates).map(([key, rate]) => (
                                <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', padding: '0.5rem', borderLeft: `3px solid ${rate.color}` }}>
                                    <div>
                                        <strong style={{ textTransform: 'capitalize' }}>{key.replace('_', ' ')}</strong>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{rate.time}</div>
                                    </div>
                                    <div style={{ color: rate.color, fontWeight: 700 }}>${rate.rate}/kWh</div>
                                </div>
                            ))}
                        </div>
                        <div className="card" style={{ padding: '1rem', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1))' }}>
                            <div style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>Annual Charging Cost Comparison:</div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                <span>Always Peak (4-9pm)</span>
                                <span style={{ color: 'var(--accent-red)' }}>${tou.example_california_pge.annual_comparison.always_peak.toLocaleString()}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                <span>Off-Peak</span>
                                <span style={{ color: 'var(--accent-orange)' }}>${tou.example_california_pge.annual_comparison.always_off_peak.toLocaleString()}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <span>Super Off-Peak (12-6am)</span>
                                <span style={{ color: 'var(--accent-green)', fontWeight: 700 }}>${tou.example_california_pge.annual_comparison.super_off_peak}</span>
                            </div>
                            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '0.5rem', marginTop: '0.5rem' }}>
                                <strong style={{ color: 'var(--accent-green)' }}>💰 Savings with smart charging: ${tou.example_california_pge.annual_comparison.savings_smart_charging.toLocaleString()}/year</strong>
                            </div>
                        </div>
                    </div>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>🆓 Free Charging Opportunities</h4>
                        {tou.free_charging_options.map((option, i) => (
                            <div key={i} className="card" style={{ padding: '0.75rem', marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between' }}>
                                <span>{option.provider}</span>
                                <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>~${option.typical_savings}/yr</span>
                            </div>
                        ))}
                        <div className="card" style={{ padding: '1rem', marginTop: '1rem', borderLeft: '3px solid var(--accent-blue)' }}>
                            <h5 style={{ marginBottom: '0.5rem' }}>💡 Smart Charging Tips</h5>
                            <ul style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', paddingLeft: '1rem', margin: 0 }}>
                                <li>Schedule charging for 12am-6am</li>
                                <li>Use Tesla/app scheduled departure</li>
                                <li>Charge to 80% daily, 100% only for trips</li>
                                <li>Slower charging = better for battery life</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            {/* Depreciation Curves */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📉 Depreciation: Which EVs Hold Value?</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="5-Year Depreciation Curves"
                        insight="Tesla Model 3 holds value like a Toyota - only 32% depreciation in 5 years. Nissan Leaf loses 62%. The difference is brand perception, battery confidence, and OTA updates keeping cars fresh."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={charts.depreciation_curves}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="year" stroke="#71717a" label={{ value: 'Years', position: 'bottom' }} />
                                <YAxis stroke="#71717a" unit="%" domain={[30, 100]} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => `${v}% of MSRP`}
                                />
                                <Legend />
                                <Line type="monotone" dataKey="tesla_model_3" name="Tesla Model 3" stroke="#22c55e" strokeWidth={3} dot={{ r: 4 }} />
                                <Line type="monotone" dataKey="avg_ev" name="Average EV" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                                <Line type="monotone" dataKey="avg_gas" name="Average Gas" stroke="#f97316" strokeWidth={2} dot={{ r: 3 }} />
                                <Line type="monotone" dataKey="nissan_leaf" name="Nissan Leaf" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>EV Depreciation Rankings (5-year)</h4>
                        {Object.entries(depreciation.ev_models).sort((a, b) => a[1].depreciation_5yr_percent - b[1].depreciation_5yr_percent).map(([key, model], i) => (
                            <div key={key} className="card" style={{ padding: '0.6rem', marginBottom: '0.4rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <div>
                                        <span style={{ fontWeight: 600 }}>{model.name}</span>
                                        <span style={{
                                            marginLeft: '0.5rem', fontSize: '0.7rem', padding: '0.15rem 0.4rem', borderRadius: '4px',
                                            background: model.depreciation_5yr_percent < 40 ? 'rgba(34, 197, 94, 0.2)' : model.depreciation_5yr_percent < 55 ? 'rgba(249, 115, 22, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                                            color: model.depreciation_5yr_percent < 40 ? '#22c55e' : model.depreciation_5yr_percent < 55 ? '#f97316' : '#ef4444'
                                        }}>{model.ranking}</span>
                                    </div>
                                    <div style={{ textAlign: 'right' }}>
                                        <span style={{ color: model.depreciation_5yr_percent < 40 ? '#22c55e' : model.depreciation_5yr_percent < 55 ? '#f97316' : '#ef4444', fontWeight: 700 }}>
                                            -{model.depreciation_5yr_percent}%
                                        </span>
                                    </div>
                                </div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                                    ${model.msrp.toLocaleString()} → ${model.value_after_5yr.toLocaleString()}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Insurance Costs */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🛡️ Insurance Cost Comparison</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="Annual Insurance Premium by Model"
                        insight="EVs cost 28% more to insure on average ($2,280 vs $1,780). Teslas are expensive due to repair costs and performance. Chevy Bolt is cheapest EV to insure. Tesla Insurance can save 20-40% for Tesla owners."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={charts.insurance_by_model} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis type="number" stroke="#71717a" tickFormatter={(v) => `$${v}`} />
                                <YAxis dataKey="model" type="category" stroke="#71717a" width={120} tick={{ fontSize: 11 }} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => `$${v.toLocaleString()}/year`}
                                />
                                <Bar dataKey="annual_premium" name="Annual Premium" radius={[0, 8, 8, 0]}>
                                    {charts.insurance_by_model.map((entry, i) => (
                                        <Cell key={i} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <div className="card" style={{ padding: '1rem', marginBottom: '1rem', background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(249, 115, 22, 0.1))' }}>
                            <h4 style={{ marginBottom: '0.75rem' }}>Why EVs Cost More to Insure</h4>
                            {insurance.why_evs_cost_more.slice(0, 4).map((reason, i) => (
                                <div key={i} style={{ fontSize: '0.85rem', marginBottom: '0.4rem', paddingLeft: '0.5rem', borderLeft: '2px solid var(--accent-orange)' }}>
                                    <strong>{reason.reason}:</strong> {reason.impact}
                                </div>
                            ))}
                        </div>
                        <div className="card" style={{ padding: '1rem', borderLeft: '3px solid var(--accent-green)' }}>
                            <h4 style={{ marginBottom: '0.75rem', color: 'var(--accent-green)' }}>💡 Ways to Save</h4>
                            {insurance.ways_to_save.map((way, i) => (
                                <div key={i} style={{ fontSize: '0.85rem', display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                                    <span>{way.method}</span>
                                    <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>{way.savings}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Repair & Maintenance */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🔧 Repair & Maintenance Costs</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <div>
                        <h4 style={{ marginBottom: '0.75rem', color: '#22c55e' }}>⚡ Common EV Repairs</h4>
                        <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                            <table style={{ width: '100%', fontSize: '0.8rem' }}>
                                <thead>
                                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                        <th style={{ textAlign: 'left', padding: '0.4rem' }}>Repair</th>
                                        <th style={{ textAlign: 'right', padding: '0.4rem' }}>Cost</th>
                                        <th style={{ textAlign: 'right', padding: '0.4rem' }}>Frequency</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {repair.common_ev_repairs.map((item, i) => (
                                        <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                                            <td style={{ padding: '0.4rem' }}>{item.repair}</td>
                                            <td style={{ textAlign: 'right', padding: '0.4rem' }}>${item.cost_low}-${item.cost_high}</td>
                                            <td style={{ textAlign: 'right', padding: '0.4rem', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>{item.frequency}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem', color: '#ef4444' }}>⛽ Common Gas Repairs</h4>
                        <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                            <table style={{ width: '100%', fontSize: '0.8rem' }}>
                                <thead>
                                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                        <th style={{ textAlign: 'left', padding: '0.4rem' }}>Repair</th>
                                        <th style={{ textAlign: 'right', padding: '0.4rem' }}>Cost</th>
                                        <th style={{ textAlign: 'right', padding: '0.4rem' }}>Frequency</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {repair.common_gas_repairs.map((item, i) => (
                                        <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                                            <td style={{ padding: '0.4rem' }}>{item.repair}</td>
                                            <td style={{ textAlign: 'right', padding: '0.4rem' }}>${item.cost_low}-${item.cost_high}</td>
                                            <td style={{ textAlign: 'right', padding: '0.4rem', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>{item.frequency}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <div style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                    {charts.maintenance_annual.map((item, i) => (
                        <div key={i} className="card" style={{ padding: '1rem', textAlign: 'center', borderTop: `3px solid ${item.color}` }}>
                            <div style={{ fontSize: '2rem', fontWeight: 700, color: item.color }}>${item.cost.toLocaleString()}</div>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{item.type} Annual Maintenance</div>
                        </div>
                    ))}
                </div>

                <div className="card" style={{ marginTop: '1rem', padding: '1rem', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(59, 130, 246, 0.1))', textAlign: 'center' }}>
                    <span style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-green)' }}>${repair.ten_year_maintenance.savings.toLocaleString()}</span>
                    <div style={{ color: 'var(--text-secondary)' }}>10-Year Maintenance Savings (EV vs Gas)</div>
                </div>
            </div>

            {/* Lease vs Buy */}
            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">📋 Lease vs Buy Analysis - Tesla Model 3 Example</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-purple)', marginBottom: '0.75rem' }}>📝 Lease (36 months)</h4>
                        <div style={{ marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Monthly</span>
                                <span style={{ fontWeight: 600 }}>${leaseBuy.example_tesla_model_3.lease.monthly_payment}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Down payment</span>
                                <span>${leaseBuy.example_tesla_model_3.lease.down_payment.toLocaleString()}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--border)', paddingTop: '0.5rem', marginTop: '0.5rem' }}>
                                <span style={{ fontWeight: 600 }}>Total 3yr cost</span>
                                <span style={{ color: 'var(--accent-purple)', fontWeight: 700 }}>${leaseBuy.example_tesla_model_3.lease.total_cost.toLocaleString()}</span>
                            </div>
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.75rem' }}>
                            {leaseBuy.example_tesla_model_3.lease.miles_allowed.toLocaleString()} miles allowed
                        </div>
                    </div>

                    <div className="card" style={{ padding: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-blue)', marginBottom: '0.75rem' }}>💳 Buy (Loan 72mo)</h4>
                        <div style={{ marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Monthly ({leaseBuy.example_tesla_model_3.buy_loan.apr}% APR)</span>
                                <span style={{ fontWeight: 600 }}>${leaseBuy.example_tesla_model_3.buy_loan.monthly_payment}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Total paid</span>
                                <span>${leaseBuy.example_tesla_model_3.buy_loan.total_loan_cost.toLocaleString()}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Value after 6yr</span>
                                <span style={{ color: 'var(--accent-green)' }}>-${leaseBuy.example_tesla_model_3.buy_loan.value_after_6yr.toLocaleString()}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--border)', paddingTop: '0.5rem', marginTop: '0.5rem' }}>
                                <span style={{ fontWeight: 600 }}>Net cost</span>
                                <span style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>${leaseBuy.example_tesla_model_3.buy_loan.net_cost.toLocaleString()}</span>
                            </div>
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--accent-green)', marginTop: '0.75rem' }}>
                            + $7,500 tax credit!
                        </div>
                    </div>

                    <div className="card" style={{ padding: '1rem' }}>
                        <h4 style={{ color: 'var(--accent-green)', marginBottom: '0.75rem' }}>💵 Buy (Cash)</h4>
                        <div style={{ marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Purchase</span>
                                <span>${leaseBuy.example_tesla_model_3.buy_cash.purchase_price.toLocaleString()}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Tax credit</span>
                                <span style={{ color: 'var(--accent-green)' }}>-${leaseBuy.example_tesla_model_3.buy_cash.minus_tax_credit.toLocaleString()}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Value after 6yr</span>
                                <span style={{ color: 'var(--accent-green)' }}>-${leaseBuy.example_tesla_model_3.buy_cash.value_after_6yr.toLocaleString()}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--border)', paddingTop: '0.5rem', marginTop: '0.5rem' }}>
                                <span style={{ fontWeight: 600 }}>True cost</span>
                                <span style={{ color: 'var(--accent-green)', fontWeight: 700 }}>${leaseBuy.example_tesla_model_3.buy_cash.true_cost_ownership.toLocaleString()}</span>
                            </div>
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--accent-green)', marginTop: '0.75rem', fontWeight: 600 }}>
                            ✅ Lowest total cost
                        </div>
                    </div>
                </div>

                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1rem' }}>
                        <h5 style={{ marginBottom: '0.5rem' }}>✅ When to Lease</h5>
                        <ul style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', paddingLeft: '1rem', margin: 0 }}>
                            {leaseBuy.when_to_lease.map((item, i) => (
                                <li key={i}>{item}</li>
                            ))}
                        </ul>
                    </div>
                    <div className="card" style={{ padding: '1rem' }}>
                        <h5 style={{ marginBottom: '0.5rem' }}>✅ When to Buy</h5>
                        <ul style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', paddingLeft: '1rem', margin: 0 }}>
                            {leaseBuy.when_to_buy.map((item, i) => (
                                <li key={i}>{item}</li>
                            ))}
                        </ul>
                    </div>
                </div>
            </div>

            {/* Total Cost of Ownership */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1))' }}>
                <h3 className="chart-title">💰 7-Year Total Cost of Ownership</h3>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.85rem' }}>
                    12,000 miles/year. Includes purchase, fuel, maintenance, insurance, registration. Minus resale value.
                </p>

                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <ChartModal
                        title="TCO Breakdown by Category"
                        insight="The Accord Hybrid wins on 7-year TCO at $57,430. Tesla Model 3 is $62,700 but beats the Camry ($69,350) thanks to fuel and maintenance savings. EVs win for high-mileage drivers."
                    >
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={charts.tco_breakdown}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                                <XAxis dataKey="category" stroke="#71717a" angle={-20} textAnchor="end" height={70} tick={{ fontSize: 10 }} />
                                <YAxis stroke="#71717a" tickFormatter={(v) => `$${v / 1000}k`} />
                                <Tooltip
                                    contentStyle={{ background: '#18181b', border: '1px solid #27272a', borderRadius: 8 }}
                                    formatter={(v) => `$${v.toLocaleString()}`}
                                />
                                <Legend />
                                <Bar dataKey="Tesla Model 3" fill="#22c55e" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="Camry" fill="#ef4444" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="Accord Hybrid" fill="#f97316" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartModal>

                    <div>
                        <h4 style={{ marginBottom: '0.75rem' }}>🏆 7-Year TCO Rankings</h4>
                        {charts.tco_total.sort((a, b) => a.total - b.total).map((item, i) => (
                            <div key={i} className="card" style={{ padding: '1rem', marginBottom: '0.5rem', borderLeft: `4px solid ${item.color}` }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <div>
                                        <span style={{ fontSize: '1.25rem', fontWeight: 700 }}>#{i + 1}</span>
                                        <span style={{ marginLeft: '0.75rem', fontWeight: 600 }}>{item.vehicle}</span>
                                    </div>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: item.color }}>
                                        ${item.total.toLocaleString()}
                                    </div>
                                </div>
                            </div>
                        ))}

                        <div className="card" style={{ padding: '1rem', marginTop: '1rem', background: 'rgba(34, 197, 94, 0.1)' }}>
                            <h5 style={{ marginBottom: '0.5rem' }}>Who Wins When?</h5>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                <div style={{ marginBottom: '0.25rem' }}><strong>High mileage (20k+/yr):</strong> EV wins big</div>
                                <div style={{ marginBottom: '0.25rem' }}><strong>Average (12k/yr):</strong> Hybrid or EV</div>
                                <div><strong>Low mileage (8k/yr):</strong> Hybrid or gas</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Bottom Line */}
            <div className="chart-container" style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.1))' }}>
                <h3 className="chart-title">🎯 The Bottom Line</h3>
                <div className="grid-2" style={{ marginTop: '1rem' }}>
                    <div className="card" style={{ padding: '1.5rem' }}>
                        <h4 style={{ color: 'var(--accent-green)', marginBottom: '0.75rem' }}>✅ EV Wins</h4>
                        <ul style={{ fontSize: '0.9rem', paddingLeft: '1rem', margin: 0 }}>
                            <li>Fuel: <strong>$1,100+ savings/year</strong> in most states</li>
                            <li>Maintenance: <strong>$800/year savings</strong></li>
                            <li>Depreciation: Tesla holds value like Toyota</li>
                            <li>Tax credit: <strong>$7,500 off purchase</strong></li>
                        </ul>
                    </div>
                    <div className="card" style={{ padding: '1.5rem' }}>
                        <h4 style={{ color: 'var(--accent-orange)', marginBottom: '0.75rem' }}>⚠️ Watch Out For</h4>
                        <ul style={{ fontSize: '0.9rem', paddingLeft: '1rem', margin: 0 }}>
                            <li>Insurance: <strong>28% higher</strong> on average</li>
                            <li>CA electricity: <strong>27¢/kWh</strong> - charge off-peak!</li>
                            <li>Some EVs depreciate fast (Leaf, Bolt)</li>
                            <li>Out-of-warranty battery: <strong>$15-25k</strong></li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    )
}
