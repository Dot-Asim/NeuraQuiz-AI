import { Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { Brain, Code, Mail } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import NotFound from './pages/NotFound';
import './App.css';

function App() {
  const location = useLocation();

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
        <header className="header">
          <NavLink to="/" style={{ textDecoration: 'none' }} className="logo-container">
            <Brain size={32} className="logo-icon" />
            <span className="logo-text">NeuraQuiz AI</span>
            <span className="badge">AI Lab Project</span>
          </NavLink>
          <nav className="nav-links">
            <NavLink 
              to="/" 
              className={({ isActive }) => `btn btn-secondary ${isActive ? 'active' : ''}`}
            >
              Home
            </NavLink>
            <NavLink 
              to="/app" 
              className={({ isActive }) => `btn btn-secondary ${isActive ? 'active' : ''}`}
            >
              Dashboard
            </NavLink>
          </nav>
        </header>

        <main className="main-content">
          <AnimatePresence mode="wait">
            <Routes location={location} key={location.pathname}>
              <Route path="/" element={
                <PageTransition>
                  <Landing />
                </PageTransition>
              } />
              <Route path="/app" element={
                <PageTransition>
                  <Dashboard />
                </PageTransition>
              } />
              <Route path="*" element={
                <PageTransition>
                  <NotFound />
                </PageTransition>
              } />
            </Routes>
          </AnimatePresence>
        </main>

        <footer className="footer">
          <div className="footer-links">
            <a href="https://github.com/Dot-Asim/NeuraQuiz-AI" target="_blank" rel="noopener noreferrer" className="flex-center gap-2">
              <Code size={18} /> GitHub Repository
            </a>
            <span>•</span>
            <a href="mailto:asimmuhammad6780@gmail.com" className="flex-center gap-2">
              <Mail size={18} /> Contact
            </a>
          </div>
          <p>Built with React, FastAPI, and CUDA-accelerated XGBoost.</p>
          <p>© {new Date().getFullYear()} NeuraQuiz AI Team. All rights reserved.</p>
        </footer>
      </div>
    </>
  );
}

// Reusable transition wrapper for routes
const PageTransition = ({ children }: { children: React.ReactNode }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="w-full"
    >
      {children}
    </motion.div>
  );
};

export default App;
