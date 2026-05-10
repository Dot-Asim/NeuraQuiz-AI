import { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, FileText, Settings2, Play, CheckCircle2, XCircle, RefreshCw, HelpCircle, AlertCircle, Zap, Activity, BookOpen, Download, Eye } from 'lucide-react';
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

  // Dynamic telemetry data
  const [latencyData, setLatencyData] = useState<{time: string, ms: number}[]>([]);
  const [avgLatency, setAvgLatency] = useState(0);

  useEffect(() => {
    if (latencyData.length > 0) {
      const sum = latencyData.reduce((acc, curr) => acc + curr.ms, 0);
      setAvgLatency(Math.round(sum / latencyData.length));
    }
  }, [latencyData]);

  const loadRandomArticle = async () => {
    try {
      setLoading(true);
      const startTime = performance.now();
      const res = await axios.get(`${API_URL}/random_article`);
      const endTime = performance.now();
      
      setArticle(res.data.article);
      setQuestion(res.data.question);
      setCorrectAnswer(res.data.correct_answer);
      
      setLatencyData(prev => [...prev.slice(-9), { time: new Date().toLocaleTimeString().slice(0,5), ms: Math.round(endTime - startTime) }]);
      toast.success('Random article loaded');
    } catch (err) {
      toast.error('Failed to load random article. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const generateQuiz = async () => {
    if (!article.trim()) {
      toast.error('Please provide a reading passage.');
      return;
    }
    try {
      setLoading(true);
      const startTime = performance.now();
      const res = await axios.post(`${API_URL}/quiz`, {
        article, question, correct_answer: correctAnswer
      });
      const endTime = performance.now();
      
      setQuiz(res.data);
      setSelectedOption(null);
      setVerificationResult(null);
      setHintsRevealed(0);
      
      // Use actual backend latency if provided, else use frontend measured
      const latency = res.data.total_latency_ms || Math.round(endTime - startTime);
      setLatencyData(prev => [...prev.slice(-9), { time: new Date().toLocaleTimeString().slice(0,5), ms: latency }]);
      toast.success('Quiz generated successfully');
    } catch (err) {
      toast.error('Failed to generate quiz. Check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  const verifyAnswer = async () => {
    if (!selectedOption) return;
    try {
      setVerifying(true);
      const optionText = quiz?.options.find((o: any) => o.label === selectedOption)?.text;
      
      const startTime = performance.now();
      const res = await axios.post(`${API_URL}/verify`, {
        article, question, selected_option: optionText
      });
      const endTime = performance.now();
      
      setVerificationResult(res.data);
      
      if (res.data.is_correct) {
        toast.success(`Correct! Model Confidence: ${(res.data.confidence*100).toFixed(1)}%`);
      } else {
        toast.error('Incorrect according to the model.');
      }
      
      const latency = res.data.latency_ms || Math.round(endTime - startTime);
      setLatencyData(prev => [...prev.slice(-9), { time: new Date().toLocaleTimeString().slice(0,5), ms: latency }]);
    } catch (err) {
      toast.error('Failed to verify answer');
    } finally {
      setVerifying(false);
    }
  };

  const reset = () => {
    setQuiz(null); setArticle(''); setQuestion(''); setCorrectAnswer('');
  };

  const revealAnswer = () => {
    if (quiz && !verificationResult) {
      setSelectedOption(quiz.correct_label);
      setVerificationResult({ is_correct: true, confidence: 1.0, latency_ms: 0 });
      toast.success('Answer revealed.');
    }
  };

  const exportCSV = () => {
    if (latencyData.length === 0) {
      toast.error('No session data to export.');
      return;
    }
    let csvContent = "data:text/csv;charset=utf-8,Time,Latency_ms\n";
    latencyData.forEach(row => {
      csvContent += `${row.time},${row.ms}\n`;
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "neuraquiz_session_log.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="w-full">
      {/* Top Stats */}
      <div className="dashboard-stats">
        <div className="card glass-panel stat-card">
          <div className="stat-icon primary"><Activity size={24} /></div>
          <div>
            <div className="stat-label">Avg System Latency</div>
            <div className="stat-value">{avgLatency > 0 ? `${avgLatency}ms` : '--'}</div>
          </div>
        </div>
        <div className="card glass-panel stat-card">
          <div className="stat-icon success"><Brain size={24} /></div>
          <div>
            <div className="stat-label">Active Models</div>
            <div className="stat-value">XGBoost + SBERT</div>
          </div>
        </div>
      </div>

      <div className="dashboard-grid two-cols mb-8">
        {/* Left Column: Configuration */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="card glass-panel h-fit">
          <div className="flex-between mb-4">
            <h2 className="card-title"><Settings2 /> Configuration</h2>
            {quiz && (
              <button onClick={reset} className="btn btn-secondary btn-sm" title="Reset Setup">
                <RefreshCw size={14} /> Reset
              </button>
            )}
          </div>
          
          <div className="input-group mb-4">
            <div className="flex-between">
              <label className="input-label">Reading Passage</label>
              <button onClick={loadRandomArticle} className="badge" disabled={loading}>
                <RefreshCw size={12} className={loading ? "animate-spin" : ""} /> Random Sample
              </button>
            </div>
            <textarea className="text-area" rows={6} placeholder="Paste context here..." value={article} onChange={e => setArticle(e.target.value)} disabled={loading || quiz !== null} />
          </div>
          
          <div className="input-group mb-4">
            <label className="input-label">Question</label>
            <input type="text" className="text-input" placeholder="Enter the question..." value={question} onChange={e => setQuestion(e.target.value)} disabled={loading || quiz !== null} />
          </div>
          
          <div className="input-group mb-6">
            <label className="input-label">Correct Answer</label>
            <input type="text" className="text-input" placeholder="Enter the correct answer..." value={correctAnswer} onChange={e => setCorrectAnswer(e.target.value)} disabled={loading || quiz !== null} />
          </div>
          
          {!quiz && (
            <button className="btn btn-primary w-full" onClick={generateQuiz} disabled={loading || !article}>
              {loading ? <RefreshCw className="animate-spin" /> : <Play />}
              {loading ? 'Processing...' : (question && correctAnswer ? 'Generate Quiz' : 'Auto-Generate Quiz')}
            </button>
          )}
        </motion.div>

        {/* Right Column: Quiz Interface or Empty State */}
        <div className="h-full">
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="card glass-panel h-full">
                <h2 className="card-title mb-6"><FileText /> Generating Quiz...</h2>
                <div className="skeleton mb-4" style={{ height: '80px', width: '100%' }}></div>
                <div className="options-container">
                  <div className="skeleton" style={{ height: '56px', width: '100%' }}></div>
                  <div className="skeleton" style={{ height: '56px', width: '100%' }}></div>
                  <div className="skeleton" style={{ height: '56px', width: '100%' }}></div>
                  <div className="skeleton" style={{ height: '56px', width: '100%' }}></div>
                </div>
              </motion.div>
            ) : !quiz ? (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="card glass-panel h-full flex-center text-center">
                <div className="mb-6 opacity-50"><BookOpen size={64} color="var(--text-muted)" /></div>
                <h3 className="text-2xl mb-2">Ready to Learn?</h3>
                <p className="text-muted max-w-md mx-auto">
                  Paste an article and a question in the configuration panel, or load a random sample to generate an intelligent quiz.
                </p>
              </motion.div>
            ) : (
              <motion.div key="quiz" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="card glass-panel h-full">
                <h2 className="card-title mb-4"><FileText /> Interactive Quiz</h2>
                
                <div className="quiz-question-box">
                  <h3 className="quiz-question-text">{quiz.question}</h3>
                  <div className="options-container">
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
                          <span className="option-label">{opt.label}</span>
                          <span>{opt.text}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {!verificationResult ? (
                  <div className="action-bar">
                    <button className="btn btn-primary" onClick={verifyAnswer} disabled={!selectedOption || verifying}>
                      {verifying ? <RefreshCw className="animate-spin" /> : <CheckCircle2 />} Verify Answer
                    </button>
                    <button className="btn btn-secondary" onClick={() => setHintsRevealed(Math.min(quiz.hints.length, hintsRevealed + 1))} disabled={hintsRevealed >= quiz.hints.length}>
                      <HelpCircle /> Reveal Hint {hintsRevealed}/{quiz.hints.length}
                    </button>
                    {hintsRevealed >= quiz.hints.length && (
                      <button className="btn btn-danger" onClick={revealAnswer}>
                        <Eye /> Reveal Answer
                      </button>
                    )}
                  </div>
                ) : null}

                {hintsRevealed > 0 && (
                  <div className="hints-container">
                    {quiz.hints.slice(0, hintsRevealed).map((hint: any, i: number) => (
                      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} key={i} className="hint-box">
                        <AlertCircle className="hint-icon" />
                        <div>
                          <div className="hint-level">Hint Level {hint.level}</div>
                          <div className="hint-text">{hint.text}</div>
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
                      <div className="stat-item"><Zap size={14} /> Latency: {verificationResult.latency_ms?.toFixed(1) || '--'}ms</div>
                      <div className="stat-item"><Brain size={14} /> Confidence: {(verificationResult.confidence * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Analytics Footer Section */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="card glass-panel w-full">
        <div className="flex-between mb-4 flex-wrap gap-4">
          <h2 className="card-title mb-0"><Activity /> Telemetry Analytics</h2>
          <button className="btn btn-secondary btn-sm" onClick={exportCSV}>
            <Download size={14} /> Export Session CSV
          </button>
        </div>
        
        <div className="dashboard-grid two-cols mb-6">
          <div className="glass-panel p-4" style={{ padding: '1rem', borderRadius: '8px', background: 'rgba(255,255,255,0.02)' }}>
             <h3 className="text-lg font-bold mb-2">Model A (Verifier)</h3>
             <div className="flex-between text-sm text-muted mb-1"><span>Accuracy</span> <span className="text-main">~62.8%</span></div>
             <div className="flex-between text-sm text-muted mb-1"><span>Macro F1</span> <span className="text-main">~54.8%</span></div>
             <div className="flex-between text-sm text-muted"><span>Precision</span> <span className="text-main">~54.8%</span></div>
          </div>
          <div className="glass-panel p-4" style={{ padding: '1rem', borderRadius: '8px', background: 'rgba(255,255,255,0.02)' }}>
             <h3 className="text-lg font-bold mb-2">Model B (Distractor)</h3>
             <div className="flex-between text-sm text-muted mb-1"><span>Accuracy</span> <span className="text-main">~99.8%</span></div>
             <div className="flex-between text-sm text-muted mb-1"><span>Precision</span> <span className="text-main">~99.7%</span></div>
             <div className="flex-between text-sm text-muted"><span>Recall</span> <span className="text-main">~99.9%</span></div>
          </div>
        </div>

        <div className="chart-container">
          {latencyData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={latencyData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                <XAxis dataKey="time" stroke="#94a3b8" tick={{fontSize: 12}} />
                <YAxis stroke="#94a3b8" tick={{fontSize: 12}} />
                <Tooltip 
                  contentStyle={{ background: '#1e1e1e', border: '1px solid #334155', borderRadius: '8px' }}
                  itemStyle={{ color: '#f8fafc' }}
                />
                <Line type="monotone" dataKey="ms" name="Latency (ms)" stroke="var(--primary)" strokeWidth={3} dot={{ fill: 'var(--primary)', r: 4 }} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex-center h-full text-muted flex-col gap-2">
              <Activity size={32} style={{ opacity: 0.5 }} />
              <p>No telemetry data yet. Run an operation to populate chart.</p>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
