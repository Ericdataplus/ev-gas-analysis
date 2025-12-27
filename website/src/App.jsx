import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Predictions from './pages/Predictions'
import Safety from './pages/Safety'
import Costs from './pages/Costs'
import SupplyChain from './pages/SupplyChain'
import Environment from './pages/Environment'
import Sources from './pages/Sources'
import MarketInsights from './pages/MarketInsights'
import UsedEVs from './pages/UsedEVs'
import HomeEnergy from './pages/HomeEnergy'
import Solar from './pages/Solar'
import Semis from './pages/Semis'
import DeepAnalysis from './pages/DeepAnalysis'
import './index.css'

function App() {
  return (
    <BrowserRouter basename="/ev-gas-analysis">
      <div className="app">
        <aside className="sidebar">
          <div className="sidebar-logo">
            <span>⚡</span>
            <span>Energy Analysis</span>
          </div>

          <nav className="sidebar-nav">
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', padding: '0.5rem 1rem', marginTop: '0.5rem' }}>OVERVIEW</div>
            <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>📊</span> Dashboard
            </NavLink>
            <NavLink to="/deep-analysis" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>🔬</span> Deep Analysis
            </NavLink>

            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', padding: '0.5rem 1rem', marginTop: '1rem' }}>VEHICLES</div>
            <NavLink to="/market" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>🏆</span> 2024 Market
            </NavLink>
            <NavLink to="/used-evs" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>🚗</span> Used EVs
            </NavLink>
            <NavLink to="/semis" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>🚛</span> Semi Trucks
            </NavLink>
            <NavLink to="/predictions" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>📈</span> ML Predictions
            </NavLink>

            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', padding: '0.5rem 1rem', marginTop: '1rem' }}>HOME ENERGY</div>
            <NavLink to="/home-energy" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>🏠</span> Home Energy
            </NavLink>
            <NavLink to="/solar" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>☀️</span> Solar
            </NavLink>

            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', padding: '0.5rem 1rem', marginTop: '1rem' }}>ANALYSIS</div>
            <NavLink to="/safety" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>🔒</span> Safety
            </NavLink>
            <NavLink to="/costs" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>💰</span> Costs
            </NavLink>
            <NavLink to="/supply-chain" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>🏭</span> Supply Chain
            </NavLink>
            <NavLink to="/environment" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>🌍</span> Environment
            </NavLink>
            <NavLink to="/sources" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>📚</span> Sources
            </NavLink>
          </nav>

          <div style={{ paddingTop: '1rem', borderTop: '1px solid var(--border-color)', marginTop: 'auto' }}>
            <a href="https://github.com/Ericdataplus/ev-gas-analysis" target="_blank" className="nav-link">
              <span>💻</span> GitHub
            </a>
          </div>
        </aside>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/deep-analysis" element={<DeepAnalysis />} />
            <Route path="/market" element={<MarketInsights />} />
            <Route path="/used-evs" element={<UsedEVs />} />
            <Route path="/semis" element={<Semis />} />
            <Route path="/home-energy" element={<HomeEnergy />} />
            <Route path="/solar" element={<Solar />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/safety" element={<Safety />} />
            <Route path="/costs" element={<Costs />} />
            <Route path="/supply-chain" element={<SupplyChain />} />
            <Route path="/environment" element={<Environment />} />
            <Route path="/sources" element={<Sources />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
