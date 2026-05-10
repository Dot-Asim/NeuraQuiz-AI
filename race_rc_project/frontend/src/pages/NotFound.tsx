import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { AlertTriangle, Home } from 'lucide-react';

export default function NotFound() {
  return (
    <motion.div 
      className="flex-center flex-col text-center"
      style={{ minHeight: '60vh', gap: '2rem' }}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
    >
      <div style={{ color: 'var(--error)', opacity: 0.8 }}>
        <AlertTriangle size={80} />
      </div>
      
      <div>
        <h1 className="text-6xl font-bold mb-4">404</h1>
        <h2 className="text-2xl mb-4">Page Not Found</h2>
        <p className="text-muted text-lg max-w-md mx-auto">
          The page you are looking for doesn't exist or has been moved to another coordinate in latent space.
        </p>
      </div>

      <Link to="/" className="btn btn-primary mt-4">
        <Home size={18} /> Return to Home
      </Link>
    </motion.div>
  );
}
