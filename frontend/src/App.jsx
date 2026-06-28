import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Sparkles, Zap, Database, BarChart3, FileText, ArrowRight, CheckCircle2, ShieldAlert } from 'lucide-react';
import './App.css';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import MainContent from './components/MainContent';
import IntroSequence from './components/IntroSequence';

function App() {
  const [activeTab, setActiveTab] = useState('papers');
  const [experiments, setExperiments] = useState([]);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sharedData, setSharedData] = useState({});
  const [showIntro, setShowIntro] = useState(true);
  const [showWelcome, setShowWelcome] = useState(true);

  useEffect(() => {
    fetchExperiments();

    const handleTabChange = (e) => {
      if (typeof e.detail === 'string') {
        setActiveTab(e.detail);
      } else {
        setActiveTab(e.detail.tab);
        setSharedData(prev => ({ ...prev, ...e.detail }));
      }
    };
    window.addEventListener('changeTab', handleTabChange);
    return () => window.removeEventListener('changeTab', handleTabChange);
  }, []);

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const fetchExperiments = async () => {
    try {
      const response = await fetch(`${API_URL}/experiments`);
      const data = await response.json();
      setExperiments(data.experiments || []);
    } catch (error) {
      console.error('Error fetching experiments:', error);
    }
  };

  const handleDeleteExperiment = async (experimentId) => {
    try {
      const response = await fetch(`${API_URL}/experiments/${experimentId}`, {
        method: 'DELETE'
      });

      let data = {};
      try {
        data = await response.json();
      } catch {
        data = {};
      }

      if (!response.ok && response.status !== 404) {
        throw new Error(data.detail || 'Failed to delete experiment');
      }

      setExperiments(prev => prev.filter(exp => exp.id !== experimentId));
      if (selectedExperiment?.id === experimentId) {
        setSelectedExperiment(null);
      }

      await fetchExperiments();
    } catch (error) {
      console.error('Error deleting experiment:', error);
      fetchExperiments();
    }
  };

  const featureIcons = [
    { Icon: FileText, label: 'Research', color: '#3b82f6' },
    { Icon: Database, label: 'Data Lab', color: '#8b5cf6' },
    { Icon: Zap, label: 'Engine', color: '#f59e0b' },
    { Icon: BarChart3, label: 'Insights', color: '#10b981' }
  ];

  return (
    <div className="app">
      <AnimatePresence>
        {showIntro && (
          <IntroSequence key="intro" onComplete={() => setShowIntro(false)} />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {!showIntro && showWelcome && (
          <motion.div
            key="welcome"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, y: -100, scale: 0.95 }}
            transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
            style={{
              position: 'fixed',
              inset: 0,
              zIndex: 100,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'var(--bg-primary)',
              gap: '0',
              overflowY: 'auto',
              padding: '2rem 1rem'
            }}
          >
            <div className="intro-grid" style={{
              position: 'absolute',
              inset: 0,
              pointerEvents: 'none'
            }} />
            <div style={{
              position: 'absolute',
              inset: 0,
              background: 'radial-gradient(ellipse at center, rgba(99, 102, 241, 0.15) 0%, transparent 70%)',
              pointerEvents: 'none'
            }} />

            <motion.div
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{
                type: 'spring',
                stiffness: 260,
                damping: 20,
                delay: 0.2,
                mass: 1
              }}
              style={{
                width: 120,
                height: 120,
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                borderRadius: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                boxShadow: '0 0 100px rgba(99, 102, 241, 0.6)',
                marginBottom: '2.5rem',
                position: 'relative'
              }}
            >
              <Cpu size={64} strokeWidth={1.5} />
              <motion.div
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.3, 0.6, 0.3]
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                style={{
                  position: 'absolute',
                  inset: -12,
                  border: '2px solid rgba(99, 102, 241, 0.4)',
                  borderRadius: '38px'
                }}
              />
            </motion.div>

            <motion.div
              initial={{ y: 30, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.5, duration: 0.6, ease: "easeOut" }}
              style={{ textAlign: 'center' }}
            >
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                marginBottom: '0.75rem'
              }}>
                <Sparkles size={18} color="#f59e0b" />
                <span style={{
                  color: '#f59e0b',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  letterSpacing: '0.15em',
                  textTransform: 'uppercase'
                }}>
                  AUTONOMOUS ML ENGINEER
                </span>
              </div>

              <motion.h1
                animate={{ y: [0, -2, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
                style={{
                  fontSize: '2.5rem',
                  fontWeight: 700,
                  letterSpacing: '-0.02em',
                  background: 'linear-gradient(to bottom, #ffffff 35%, #c7d2fe 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  marginBottom: '0.9rem',
                  lineHeight: 1.2,
                  padding: '0 20px'
                }}
              >
                ML-Agent
              </motion.h1>
              <p style={{
                color: 'var(--text-primary)',
                fontSize: '1.1rem',
                letterSpacing: '0.01em',
                maxWidth: '820px',
                margin: '0 auto 2.5rem',
                lineHeight: 1.7,
                fontWeight: 500,
                opacity: 0.95
              }}>
                an autonomous ML engineer that analyzes data, researches papers, trains models, and ships ML workflows
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1, duration: 0.5 }}
              style={{ display: 'flex', gap: '0.75rem', marginBottom: '2.5rem' }}
            >
              {featureIcons.map(({ Icon, label, color }, idx) => (
                <motion.div
                  key={label}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1.1 + idx * 0.12, duration: 0.4 }}
                  whileHover={{ scale: 1.1, y: -4 }}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '0.5rem',
                    padding: '0.75rem 1.25rem',
                    backgroundColor: 'rgba(255,255,255,0.03)',
                    borderRadius: '16px',
                    border: '1px solid rgba(255,255,255,0.06)'
                  }}
                >
                  <div style={{
                    width: 32,
                    height: 32,
                    borderRadius: '10px',
                    backgroundColor: `${color}20`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <Icon size={18} color={color} strokeWidth={2} />
                  </div>
                  <span style={{
                    fontSize: '0.75rem',
                    color: 'var(--text-secondary)',
                    fontWeight: 500
                  }}>
                    {label}
                  </span>
                </motion.div>
              ))}
            </motion.div>

            {/* How to use section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.4, duration: 0.6 }}
              style={{
                background: 'rgba(30, 41, 59, 0.4)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: '24px',
                padding: '2rem',
                maxWidth: '800px',
                width: '100%',
                marginBottom: '2.5rem',
                backdropFilter: 'blur(20px)',
                boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
                textAlign: 'left'
              }}
            >
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'white', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <CheckCircle2 size={22} color="#10b981" />
                How to use it
              </h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', gap: '1rem', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.5 }}>
                  <div style={{ background: 'rgba(99, 102, 241, 0.1)', color: '#8b5cf6', width: 28, height: 28, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontWeight: 'bold' }}>1</div>
                  <span>Start by researching ML papers or uploading your dataset in the Data Lab.</span>
                </div>
                <div style={{ display: 'flex', gap: '1rem', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.5 }}>
                  <div style={{ background: 'rgba(99, 102, 241, 0.1)', color: '#8b5cf6', width: 28, height: 28, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontWeight: 'bold' }}>2</div>
                  <span>Configure your task and let the agent autonomously clean data and train models.</span>
                </div>
                <div style={{ display: 'flex', gap: '1rem', color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.5 }}>
                  <div style={{ background: 'rgba(99, 102, 241, 0.1)', color: '#8b5cf6', width: 28, height: 28, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontWeight: 'bold' }}>3</div>
                  <span>Review comprehensive insights, evaluate metrics, and export your production-ready model.</span>
                </div>
              </div>

              <div style={{
                background: 'rgba(245, 158, 11, 0.05)',
                border: '1px solid rgba(245, 158, 11, 0.2)',
                borderRadius: '16px',
                padding: '1.25rem',
                display: 'flex',
                gap: '1rem',
                alignItems: 'flex-start'
              }}>
                <ShieldAlert size={22} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
                <div>
                  <h3 style={{ color: '#f59e0b', fontSize: '1rem', fontWeight: 600, margin: '0 0 0.35rem 0' }}>Important Notice</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', margin: 0, lineHeight: 1.6 }}>
                    AI can make mistakes. Always cross-verify the generated models, data preprocessing steps, and insights before using them in production.
                  </p>
                </div>
              </div>
            </motion.div>

            <motion.button
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 1.8, duration: 0.4 }}
              whileHover={{ scale: 1.05, boxShadow: '0 0 30px rgba(99, 102, 241, 0.5)' }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowWelcome(false)}
              style={{
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                color: 'white',
                border: 'none',
                padding: '1rem 2.5rem',
                borderRadius: '40px',
                fontSize: '1.1rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                boxShadow: '0 10px 25px rgba(99, 102, 241, 0.3)'
              }}
            >
              Launch Workspace
              <ArrowRight size={20} />
            </motion.button>

          </motion.div>
        )}
      </AnimatePresence>

      <Header />
      <div className="app-container">
        <Sidebar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          experiments={experiments}
          selectedExperiment={selectedExperiment}
          setSelectedExperiment={setSelectedExperiment}
          onDeleteExperiment={handleDeleteExperiment}
        />

        <MainContent
          activeTab={activeTab}
          experiments={experiments}
          selectedExperiment={selectedExperiment}
          fetchExperiments={fetchExperiments}
          sharedData={sharedData}
        />
      </div>
    </div>
  );
}

export default App;
