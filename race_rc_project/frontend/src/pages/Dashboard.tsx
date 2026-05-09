import { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, FileText, Settings2, Play, CheckCircle2, XCircle, RefreshCw, HelpCircle, AlertCircle, Zap, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import toast from 'react-hot-toast';

const API_URL = 'http://localhost:8000/api';

export default function Dashboard() {
  const [article, setArticle] = useState('');
  const [question, setQuestion] = useState('');
  const [correctAnswer, setCorrectAnswer] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [quiz, setQuiz] = useState<any>(null);
  
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [verificationResult, setVerificationResult] = useState<any>(null);
  const [verifying, setVerifying] = useState(false);
  
  const [hintsRevealed, setHintsRevealed] = useState(0);

  // Mock telemetry data for the chart
  const [latencyData, setLatencyData] = useState([
    { time: '10:00', ms: 120 }, { time: '10:01', ms: 115 }, { time: '10:02', ms: 130 }
  ]);

  const loadRandomArticle = async () => {
    try {
      setLoading(true);
      const res = await axios.get(`${API_URL}/random_article`);
      setArticle(res.data.article);
      setQuestion(res.data.question);
      setCorrectAnswer(res.data.correct_answer);
      toast.success('Random article loaded successfully');
    } catch (err) {
      toast.error('Failed to load random article');
    } finally {
      setLoading(false);
    }
  };

  const generateQuiz = async () => {
    if (!article.trim() || !question.trim() || !correctAnswer.trim()) {
      toast.error('Please fill in all fields.');
      return;
    }
    try {
      setLoading(true);
      const res = await axios.post(`${API_URL}/quiz`, {
        article, question, correct_answer: correctAnswer
      });
      setQuiz(res.data);
      setSelectedOption(null);
      setVerificationResult(null);
      setHintsRevealed(0);
      
      setLatencyData(prev => [...prev.slice(-4), { time: new Date().toLocaleTimeString().slice(0,5), ms: res.data.total_latency_ms }]);
      toast.success('Quiz engine initialized');
    } catch (err) {
      toast.error('Failed to generate quiz');
    } finally {
      setLoading(false);
    }
  };

  const verifyAnswer = async () => {
    if (!selectedOption) return;
    try {
      setVerifying(true);
      const optionText = quiz?.options.find((o: any) => o.label === selectedOption)?.text;
      const res = await axios.post(`${API_URL}/verify`, {
        article, question, selected_option: optionText
      });
      setVerificationResult(res.data);
      
      if (res.data.is_correct) {
        toast.success(`Correct! Model Confidence: ${(res.data.confidence*100).toFixed(1)}%`);
      } else {
        toast.error('Incorrect according to the model.');
      }
      
      setLatencyData(prev => [...prev.slice(-4), { time: new Date().toLocaleTimeString().slice(0,5), ms: res.data.latency_ms }]);
    } catch (err) {
      toast.error('Failed to verify answer');
    } finally {
      setVerifying(false);
    }
  };

  const reset = () => {
    setQuiz(null); setArticle(''); setQuestion(''); setCorrectAnswer('');
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '2rem' }}>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem', marginBottom: '1rem' }}>
        <div className="card glass-panel" style={{ padding: '1.5rem', flexDirection: 'row', alignItems: 'center', gap: '1rem' }}>
          <div style={{ background: 'rgba(99,102,241,0.1)', padding: '1rem', borderRadius: '12px' }}><Activity color="var(--primary)" /></div>
          <div><div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textTransform: 'uppercase' }}>Avg Latency</div><div style={{ fontSize: '1.5rem', fontWeight: 700 }}>124ms</div></div>
        </div>
        <div className="card glass-panel" style={{ padding: '1.5rem', flexDirection: 'row', alignItems: 'center', gap: '1rem' }}>
          <div style={{ background: 'rgba(16,185,129,0.1)', padding: '1rem', borderRadius: '12px' }}><Brain color="var(--success)" /></div>
          <div><div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textTransform: 'uppercase' }}>Models Active</div><div style={{ fontSize: '1.5rem', fontWeight: 700 }}>XGBoost + SBERT</div></div>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {!quiz ? (
          <motion.div key="setup" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, x: -20 }} className="card glass-panel">
            <h2 className="card-title"><Settings2 /> Configuration Panel</h2>
            <div className="input-group">
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <label className="input-label">Reading Passage</label>
                <button onClick={loadRandomArticle} className="badge" style={{ cursor: 'pointer', background: 'rgba(99,102,241,0.2)' }}>
                  <RefreshCw size={12} style={{ display: 'inline', marginRight: 4 }}/> Random Sample
                </button>
              </div>
              <textarea className="text-area" rows={6} placeholder="Paste context here..." value={article} onChange={e => setArticle(e.target.value)} />
            </div>
            <div className="input-group">
              <label className="input-label">Question</label>
              <input type="text" className="text-input" placeholder="Enter the question..." value={question} onChange={e => setQuestion(e.target.value)} />
            </div>
            <div className="input-group">
              <label className="input-label">Correct Answer</label>
              <input type="text" className="text-input" placeholder="Enter the correct answer..." value={correctAnswer} onChange={e => setCorrectAnswer(e.target.value)} />
            </div>
            <button className="btn btn-primary" onClick={generateQuiz} disabled={loading} style={{ marginTop: '1rem' }}>
              {loading ? <RefreshCw className="animate-spin" /> : <Play />}
              {loading ? 'Initializing Engine...' : 'Launch Quiz Engine'}
            </button>
          </motion.div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '2rem' }}>
            <motion.div key="quiz" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="card glass-panel">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <h2 className="card-title"><FileText /> AI Generated Quiz</h2>
                <button onClick={reset} className="btn btn-secondary" style={{ padding: '0.5rem 1rem' }}><RefreshCw size={16} /> New Setup</button>
              </div>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.5rem', borderRadius: '8px', borderLeft: '3px solid var(--primary)' }}>
                <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem' }}>{quiz.question}</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {quiz.options.map((opt: any) => {
                    let className = 'option-btn';
                    if (verificationResult) {
                      if (opt.is_correct) className += ' correct';
                      else if (selectedOption === opt.label) className += ' incorrect';
                    } else if (selectedOption === opt.label) {
                      className += ' selected';
                    }
                    return (
                      <button key={opt.label} className={className} onClick={() => !verificationResult && setSelectedOption(opt.label)} disabled={!!verificationResult}>
                        <span className="option-label">{opt.label}</span><span>{opt.text}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {!verificationResult ? (
                <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
                  <button className="btn btn-primary" style={{ flex: 1 }} onClick={verifyAnswer} disabled={!selectedOption || verifying}>
                    {verifying ? <RefreshCw className="animate-spin" /> : <CheckCircle2 />} Verify Answer
                  </button>
                  <button className="btn btn-secondary" onClick={() => setHintsRevealed(Math.min(quiz.hints.length, hintsRevealed + 1))} disabled={hintsRevealed >= quiz.hints.length}>
                    <HelpCircle /> Reveal Hint {hintsRevealed}/{quiz.hints.length}
                  </button>
                </div>
              ) : null}

              {hintsRevealed > 0 && (
                <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {quiz.hints.slice(0, hintsRevealed).map((hint: any, i: number) => (
                    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} key={i} style={{ padding: '1rem', background: 'rgba(236, 72, 153, 0.1)', border: '1px solid rgba(236, 72, 153, 0.3)', borderRadius: '8px', display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
                      <AlertCircle style={{ color: 'var(--secondary)' }} />
                      <div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.2rem' }}>Hint Level {hint.level}</div>
                        <div style={{ fontSize: '0.95rem' }}>{hint.text}</div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}

              {verificationResult && (
                <div className={`result-banner ${verificationResult.is_correct ? 'success' : 'error'}`}>
                  <div className="result-title">
                    {verificationResult.is_correct ? <CheckCircle2 size={24} /> : <XCircle size={24} />}
                    {verificationResult.is_correct ? 'Correct! Model Confirmed.' : 'Incorrect. The model disagreed.'}
                  </div>
                  <div className="stat-row">
                    <div className="stat-item"><Zap size={14} /> Latency: {verificationResult.latency_ms.toFixed(1)}ms</div>
                    <div className="stat-item"><Brain size={14} /> Confidence: {(verificationResult.confidence * 100).toFixed(1)}%</div>
                  </div>
                </div>
              )}
            </motion.div>
            
            {/* Analytics Dashboard */}
            <div className="card glass-panel">
              <h2 className="card-title" style={{ marginBottom: '1rem' }}><Activity /> Telemetry Analytics</h2>
              <div style={{ height: 300, width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={latencyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="time" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip contentStyle={{ background: '#1e1e1e', border: '1px solid #334155' }} />
                    <Line type="monotone" dataKey="ms" stroke="var(--primary)" strokeWidth={3} dot={{ fill: 'var(--primary)', r: 6 }} activeDot={{ r: 8 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
