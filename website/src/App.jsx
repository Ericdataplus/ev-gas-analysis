import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Predictions from './pages/Predictions'
import Safety from './pages/Safety'
import Costs from './pages/Costs'
import SupplyChain from './pages/SupplyChain'
import Environment from './pages/Environment'
import Sources from './pages/Sources'
import './index.css'

function App() {
  return (
    <BrowserRouter basename="/ev-gas-analysis">
      <div className="app">
        <aside className="sidebar">
          <div className="sidebar-logo">
            <span>🚗⚡</span>
            <span>EV Analysis</span>
          </div>

          <nav className="sidebar-nav">
            <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>📊</span> Dashboard
            </NavLink>
            <NavLink to="/predictions" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <span>📈</span> ML Predictions
            </NavLink>
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

          <div style={{ marginTop: 'auto', paddingTop: '2rem', borderTop: '1px solid var(--border-color)', marginTop: '2rem' }}>
            <a href="https://github.com/Ericdataplus/ev-gas-analysis" target="_blank" className="nav-link">
              <span>💻</span> GitHub
            </a>
          </div>
        </aside>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
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
