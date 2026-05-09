import { Routes, Route, Link } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { Brain } from 'lucide-react';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import './App.css';

function App() {
  return (
    <>
      <Toaster 
        position="top-right" 
        toastOptions={{
          style: {
            background: '#1e1e1e',
            color: '#f8fafc',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)',
          }
        }} 
      />
      <div className="app-container">
        <header className="header" style={{ marginBottom: '1.5rem' }}>
          <Link to="/" style={{ textDecoration: 'none' }} className="logo-container">
            <Brain size={32} className="logo-icon" />
            <span className="logo-text">NeuraQuiz AI</span>
            <span className="badge">SAAS Edition</span>
          </Link>
          <nav className="nav-links">
            <Link to="/" className="btn btn-secondary">Home</Link>
            <Link to="/app" className="btn btn-primary">Dashboard</Link>
          </nav>
        </header>

        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/app" element={<Dashboard />} />
        </Routes>
      </div>
    </>
  );
}

export default App;
