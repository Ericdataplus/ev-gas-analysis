export default function Sources() {
    const sources = [
        {
            category: "EV Sales & Adoption",
            icon: "📊",
            items: [
                { name: "IEA Global EV Outlook 2024", url: "https://www.iea.org/reports/global-ev-outlook-2024", data: "Global EV sales, market share" },
                { name: "BloombergNEF Electric Vehicle Outlook", url: "https://about.bnef.com/electric-vehicle-outlook/", data: "EV adoption forecasts, battery prices" },
                { name: "US DOE Alternative Fuels Data Center", url: "https://afdc.energy.gov/", data: "US EV registrations, charging stations" },
            ]
        },
        {
            category: "Battery Data",
            icon: "🔋",
            items: [
                { name: "BloombergNEF Battery Price Survey", url: "https://about.bnef.com/blog/lithium-ion-battery-pack-prices-hit-record-low-of-139-kwh/", data: "Battery pack prices $/kWh" },
                { name: "Geotab EV Battery Degradation Study", url: "https://www.geotab.com/blog/ev-battery-health/", data: "Battery degradation rates (6,000+ EVs)" },
            ]
        },
        {
            category: "Safety Statistics",
            icon: "🔒",
            items: [
                { name: "NHTSA", url: "https://www.nhtsa.gov/", data: "Crash statistics, fatality rates" },
                { name: "AutoinsuranceEZ Fire Study", url: "https://www.autoinsuranceez.com/gas-vs-electric-car-fires/", data: "Vehicle fire rates per 100K" },
                { name: "IIHS", url: "https://www.iihs.org/", data: "Injury claim rates, crash tests" },
                { name: "Tesla Safety Reports", url: "https://www.tesla.com/VehicleSafetyReport", data: "Autopilot crash rates" },
            ]
        },
        {
            category: "Environmental",
            icon: "🌍",
            items: [
                { name: "EPA Green Vehicles", url: "https://www.epa.gov/greenvehicles/", data: "Vehicle emissions data" },
                { name: "Argonne GREET Model", url: "https://greet.anl.gov/", data: "Lifecycle emissions analysis" },
                { name: "IEA Electricity Standards", url: "https://www.iea.org/data-and-statistics", data: "Grid emissions by country" },
            ]
        },
        {
            category: "Supply Chain",
            icon: "🏭",
            items: [
                { name: "Cobalt Institute", url: "https://www.cobaltinstitute.org/", data: "Cobalt production & demand" },
                { name: "USGS Mineral Summaries", url: "https://www.usgs.gov/centers/national-minerals-information-center/mineral-commodity-summaries", data: "Lithium, nickel reserves" },
                { name: "IEA Critical Minerals", url: "https://www.iea.org/reports/the-role-of-critical-minerals-in-clean-energy-transitions", data: "EV material requirements" },
            ]
        },
        {
            category: "Costs",
            icon: "💰",
            items: [
                { name: "AAA Fuel Prices", url: "https://gasprices.aaa.com/", data: "Gas price data" },
                { name: "Consumer Reports", url: "https://www.consumerreports.org/", data: "Maintenance cost studies" },
                { name: "iSeeCars Depreciation", url: "https://www.iseecars.com/cars-that-hold-their-value-study", data: "Vehicle depreciation rates" },
            ]
        },
        {
            category: "Transportation",
            icon: "✈️",
            items: [
                { name: "IATA", url: "https://www.iata.org/", data: "Airline fuel & emissions" },
                { name: "IMO", url: "https://www.imo.org/", data: "Shipping emissions" },
                { name: "ICCT", url: "https://theicct.org/", data: "Airline efficiency rankings" },
            ]
        },
        {
            category: "Autonomous Vehicles",
            icon: "🤖",
            items: [
                { name: "Waymo Safety Reports", url: "https://waymo.com/safety/", data: "Robotaxi operations" },
                { name: "Grand View Research", url: "https://www.grandviewresearch.com/industry-analysis/autonomous-vehicles-market", data: "AV market projections" },
            ]
        },
    ]

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">📚 Data Sources & Citations</h1>
                <p className="page-subtitle">All data sources used in this analysis for reproducibility</p>
            </header>

            <div className="chart-container" style={{ marginBottom: '1.5rem' }}>
                <p style={{ color: 'var(--text-secondary)' }}>
                    This analysis was conducted in <strong>December 2024</strong>. All data sources are publicly available
                    for verification and reproducibility.
                </p>
            </div>

            <div className="grid-2">
                {sources.map((category, i) => (
                    <div className="chart-container" key={i}>
                        <h3 className="chart-title">{category.icon} {category.category}</h3>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
                            {category.items.map((item, j) => (
                                <div key={j} className="card" style={{ padding: '1rem' }}>
                                    <a href={item.url} target="_blank" rel="noopener noreferrer"
                                        style={{ fontWeight: 600, fontSize: '0.95rem' }}>
                                        {item.name} ↗
                                    </a>
                                    <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.25rem' }}>
                                        {item.data}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            <div className="chart-container" style={{ marginTop: '1.5rem' }}>
                <h3 className="chart-title">🔄 How to Reproduce</h3>
                <ol style={{ color: 'var(--text-secondary)', paddingLeft: '1.5rem', marginTop: '1rem' }}>
                    <li style={{ marginBottom: '0.5rem' }}>Clone the repository from GitHub</li>
                    <li style={{ marginBottom: '0.5rem' }}>Visit each source URL above to download raw data</li>
                    <li style={{ marginBottom: '0.5rem' }}>Run analysis scripts in <code style={{ background: 'var(--bg-hover)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>scripts/</code></li>
                    <li style={{ marginBottom: '0.5rem' }}>Update <code style={{ background: 'var(--bg-hover)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>website/src/data/insights.json</code> with new data</li>
                    <li>Push to GitHub for automatic deployment</li>
                </ol>
            </div>
        </div>
    )
}
