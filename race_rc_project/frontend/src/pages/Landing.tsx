import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Sparkles, Zap, Shield, CheckCircle2 } from 'lucide-react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Points, PointMaterial } from '@react-three/drei';
import * as random from 'maath/random/dist/maath-random.esm';

// 3D Particles Background
function ParticleField(props: any) {
  const ref = useRef<any>();
  const [sphere] = useState(() => random.inSphere(new Float32Array(5000), { radius: 1.5 }));
  
  useFrame((state, delta) => {
    if (ref.current) {
      ref.current.rotation.x -= delta / 10;
      ref.current.rotation.y -= delta / 15;
    }
  });

  return (
    <group rotation={[0, 0, Math.PI / 4]}>
      <Points ref={ref} positions={sphere} stride={3} frustumCulled={false} {...props}>
        <PointMaterial transparent color="#6366f1" size={0.005} sizeAttenuation={true} depthWrite={false} />
      </Points>
    </group>
  );
}

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
  const [isAnnual, setIsAnnual] = useState(true);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4rem' }}>
      {/* Hero Section */}
      <section style={{ position: 'relative', height: '60vh', display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: -1, opacity: 0.6 }}>
          <Canvas camera={{ position: [0, 0, 1] }}>
            <ParticleField />
          </Canvas>
        </div>
        
        <div style={{ zIndex: 1, maxWidth: '800px' }}>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
            <div className="badge" style={{ marginBottom: '1rem', display: 'inline-block' }}>
              <Sparkles size={12} style={{ display: 'inline', marginRight: 4 }} /> 
              NeuraQuiz 2.0 is Live
            </div>
            <h1 style={{ fontSize: '3.5rem', fontWeight: 800, marginBottom: '1.5rem', background: 'linear-gradient(135deg, #fff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', minHeight: '120px' }}>
              {headline}<span style={{ animation: 'blink 1s step-end infinite' }}>|</span>
            </h1>
            <p style={{ fontSize: '1.25rem', color: 'var(--text-muted)', marginBottom: '2.5rem' }}>
              Powered by advanced XGBoost and SBERT models on a CUDA backend. Generate distractors, extract hints, and verify answers instantly.
            </p>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
              <Link to="/app" className="btn btn-primary" style={{ padding: '1rem 2rem', fontSize: '1.1rem' }}>
                Start For Free <Zap size={18} style={{ marginLeft: 8 }} />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
        {[
          { icon: <Zap color="var(--primary)" />, title: 'Lightning Fast', desc: 'CUDA-accelerated backend generates quizzes in milliseconds.' },
          { icon: <Brain color="var(--success)" />, title: 'Deep Intelligence', desc: 'SBERT models analyze semantic meaning to verify correct answers.' },
          { icon: <Shield color="var(--secondary)" />, title: 'Enterprise Grade', desc: 'Secure, reliable API endpoints ready for high-volume integration.' }
        ].map((feat, i) => (
          <motion.div key={i} className="card glass-panel" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }}>
            <div style={{ width: 48, height: 48, borderRadius: 12, background: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
              {feat.icon}
            </div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>{feat.title}</h3>
            <p style={{ color: 'var(--text-muted)' }}>{feat.desc}</p>
          </motion.div>
        ))}
      </section>

      {/* Pricing */}
      <section style={{ textAlign: 'center', marginTop: '2rem' }}>
        <h2 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>Simple, Transparent Pricing</h2>
        <div style={{ display: 'inline-flex', background: 'rgba(255,255,255,0.05)', padding: '0.25rem', borderRadius: '24px', marginBottom: '3rem' }}>
          <button className={`btn ${isAnnual ? 'btn-primary' : ''}`} style={{ borderRadius: '20px', padding: '0.5rem 1.5rem' }} onClick={() => setIsAnnual(true)}>Annual (-20%)</button>
          <button className={`btn ${!isAnnual ? 'btn-primary' : ''}`} style={{ borderRadius: '20px', padding: '0.5rem 1.5rem', background: !isAnnual ? 'var(--primary)' : 'transparent' }} onClick={() => setIsAnnual(false)}>Monthly</button>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', textAlign: 'left' }}>
          {/* Free Tier */}
          <div className="card glass-panel" style={{ opacity: 0.9 }}>
            <h3>Hobby</h3>
            <div style={{ fontSize: '2.5rem', fontWeight: 700, margin: '1rem 0' }}>$0<span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>/mo</span></div>
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '2rem', flex: 1 }}>
              <li style={{ display: 'flex', gap: '0.5rem' }}><CheckCircle2 size={18} color="var(--text-muted)" /> 100 queries / day</li>
              <li style={{ display: 'flex', gap: '0.5rem' }}><CheckCircle2 size={18} color="var(--text-muted)" /> Basic TF-IDF generation</li>
            </ul>
            <Link to="/app" className="btn btn-secondary">Get Started</Link>
          </div>
          {/* Pro Tier */}
          <div className="card glass-panel" style={{ position: 'relative', border: '1px solid var(--primary)', boxShadow: '0 0 30px rgba(99,102,241,0.15)' }}>
            <div style={{ position: 'absolute', top: -12, left: '50%', transform: 'translateX(-50%)', background: 'var(--primary)', color: '#fff', padding: '4px 12px', borderRadius: 12, fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.05em' }}>MOST POPULAR</div>
            <h3>Pro</h3>
            <div style={{ fontSize: '2.5rem', fontWeight: 700, margin: '1rem 0' }}>${isAnnual ? '29' : '39'}<span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>/mo</span></div>
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '2rem', flex: 1 }}>
              <li style={{ display: 'flex', gap: '0.5rem' }}><CheckCircle2 size={18} color="var(--primary)" /> Unlimited queries</li>
              <li style={{ display: 'flex', gap: '0.5rem' }}><CheckCircle2 size={18} color="var(--primary)" /> SBERT deep semantics</li>
              <li style={{ display: 'flex', gap: '0.5rem' }}><CheckCircle2 size={18} color="var(--primary)" /> API Access</li>
            </ul>
            <Link to="/app" className="btn btn-primary">Upgrade to Pro</Link>
          </div>
        </div>
      </section>

      <style>{`
        @keyframes blink { 50% { opacity: 0; } }
      `}</style>
    </div>
  );
}
