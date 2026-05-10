import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Sparkles, Zap, Shield, Brain, ArrowRight, BookOpen, Layers } from 'lucide-react';

// Typewriter effect hook
const useTypewriter = (text: string, speed: number = 50) => {
  const [displayText, setDisplayText] = useState('');
  
  useEffect(() => {
    let i = 0;
    const typingInterval = setInterval(() => {
      if (i < text.length) {
        setDisplayText(prev => prev + text.charAt(i));
        i++;
      } else {
        clearInterval(typingInterval);
      }
    }, speed);
    return () => clearInterval(typingInterval);
  }, [text, speed]);

  return displayText;
};

export default function Landing() {
  const headline = useTypewriter("Transform any text into an intelligent quiz.");

  return (
    <div className="flex-center" style={{ flexDirection: 'column', gap: '4rem' }}>
      {/* Hero Section */}
      <section className="flex-center" style={{ minHeight: '60vh', textAlign: 'center', position: 'relative' }}>
        <div style={{ maxWidth: '800px', zIndex: 1 }}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
            <div className="badge mb-4">
              <Sparkles size={14} /> NeuraQuiz 2.0 is Live
            </div>
            
            <h1 className="text-5xl font-bold mb-6 text-gradient" style={{ minHeight: '120px', lineHeight: 1.2 }}>
              {headline}<span style={{ animation: 'blink 1s step-end infinite' }}>|</span>
            </h1>
            
            <p className="text-xl text-muted mb-8" style={{ lineHeight: 1.6 }}>
              Powered by advanced XGBoost and SBERT models on a CUDA backend. Generate distractors, extract hints, and verify answers instantly.
            </p>
            
            <div className="flex-center gap-4">
              <Link to="/app" className="btn btn-primary" style={{ padding: '1rem 2rem', fontSize: '1.1rem' }}>
                Launch App <ArrowRight size={20} />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', width: '100%' }}>
        {[
          { icon: <Zap size={24} color="var(--primary)" />, title: 'Lightning Fast', desc: 'CUDA-accelerated backend generates quizzes in milliseconds.' },
          { icon: <Brain size={24} color="var(--success)" />, title: 'Deep Intelligence', desc: 'SBERT models analyze semantic meaning to verify correct answers.' },
          { icon: <Shield size={24} color="var(--secondary)" />, title: 'Robust Verification', desc: 'Soft-voting ensemble of 9 ML models ensures high confidence.' }
        ].map((feat, i) => (
          <motion.div key={i} className="card glass-panel feature-card" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }}>
            <div className="feature-icon">
              {feat.icon}
            </div>
            <h3 className="text-xl mb-2">{feat.title}</h3>
            <p className="text-muted">{feat.desc}</p>
          </motion.div>
        ))}
      </section>

      {/* How It Works (Replaces Pricing) */}
      <section style={{ width: '100%', marginTop: '2rem', marginBottom: '4rem' }}>
        <div className="text-center mb-8">
          <h2 className="text-4xl font-bold mb-4">How It Works</h2>
          <p className="text-muted text-lg">A seamless pipeline from raw text to interactive learning.</p>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem' }}>
          {[
            { step: '01', icon: <BookOpen size={32} />, title: 'Provide Context', desc: 'Input any reading passage or load a random sample from the RACE dataset.' },
            { step: '02', icon: <Layers size={32} />, title: 'AI Generation', desc: 'Our dual-model pipeline creates plausible distractors and progressive hints.' },
            { step: '03', icon: <Brain size={32} />, title: 'Interactive Quiz', desc: 'Test your knowledge. Our models verify your answer and provide confidence scores.' }
          ].map((item, i) => (
            <motion.div key={i} className="card glass-panel" initial={{ opacity: 0, scale: 0.95 }} whileInView={{ opacity: 1, scale: 1 }} viewport={{ once: true }} transition={{ delay: i * 0.15 }}>
              <div className="text-5xl font-bold text-muted" style={{ opacity: 0.2, marginBottom: '-20px' }}>{item.step}</div>
              <div style={{ color: 'var(--primary)', marginBottom: '1rem' }}>{item.icon}</div>
              <h3 className="text-xl mb-2">{item.title}</h3>
              <p className="text-muted">{item.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <style>{`
        @keyframes blink { 50% { opacity: 0; } }
      `}</style>
    </div>
  );
}
