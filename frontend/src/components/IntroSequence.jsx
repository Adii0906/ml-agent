import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu } from 'lucide-react';

/*
 * Cinematic boot intro. A short sequence of words/lines fade in and out,
 * then the brand reveals and we hand off to the welcome screen.
 */
const PHRASES = [
  'Research.',
  'Analyze.',
  'Train.',
  'Deploy.',
];

function IntroSequence({ onComplete }) {
  // phase: 'words' -> cycling the verbs, 'brand' -> logo reveal, then done
  const [phase, setPhase] = useState('words');
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (phase === 'words') {
      if (index < PHRASES.length) {
        const t = setTimeout(() => setIndex((i) => i + 1), 1100);
        return () => clearTimeout(t);
      }
      const t = setTimeout(() => setPhase('brand'), 400);
      return () => clearTimeout(t);
    }
    if (phase === 'brand') {
      const t = setTimeout(() => onComplete?.(), 2200);
      return () => clearTimeout(t);
    }
  }, [phase, index, onComplete]);

  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, filter: 'blur(8px)' }}
      transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 200,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg-primary)',
        overflow: 'hidden',
      }}
    >
      {/* Animated grid + ambient glow */}
      <div className="intro-grid" style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} />
      <motion.div
        animate={{ opacity: [0.35, 0.7, 0.35], scale: [1, 1.1, 1] }}
        transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          position: 'absolute',
          width: 520,
          height: 520,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(99,102,241,0.22) 0%, transparent 65%)',
          filter: 'blur(20px)',
          pointerEvents: 'none',
        }}
      />

      {/* Skip button */}
      <motion.button
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.5 }}
        whileHover={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        onClick={() => onComplete?.()}
        style={{
          position: 'absolute',
          bottom: '2rem',
          right: '2rem',
          background: 'transparent',
          border: '1px solid var(--border-color)',
          color: 'var(--text-secondary)',
          padding: '0.45rem 1.1rem',
          borderRadius: '20px',
          fontSize: '0.8rem',
          fontWeight: 500,
          letterSpacing: '0.05em',
          cursor: 'pointer',
        }}
      >
        Skip intro
      </motion.button>

      <AnimatePresence mode="wait">
        {phase === 'words' && (
          <motion.div
            key="words"
            exit={{ opacity: 0, filter: 'blur(8px)' }}
            transition={{ duration: 0.4, ease: 'easeInOut' }}
            style={{
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '100%',
              minHeight: '1.4em',
              zIndex: 1,
            }}
          >
            {/* Crossfade: incoming word fades in while the outgoing one fades out */}
            <AnimatePresence>
              <motion.h1
                key={Math.min(index, PHRASES.length - 1)}
                initial={{ opacity: 0, y: 18, filter: 'blur(12px)' }}
                animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                exit={{ opacity: 0, y: -18, filter: 'blur(12px)' }}
                transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                style={{
                  position: 'absolute',
                  fontSize: 'clamp(2.6rem, 9vw, 5.5rem)',
                  fontWeight: 800,
                  letterSpacing: '-0.03em',
                  background: 'linear-gradient(120deg, #ffffff 20%, #a5b4fc 60%, #c4b5fd 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  textAlign: 'center',
                  whiteSpace: 'nowrap',
                  margin: 0,
                  willChange: 'transform, opacity, filter',
                }}
              >
                {PHRASES[Math.min(index, PHRASES.length - 1)]}
              </motion.h1>
            </AnimatePresence>
          </motion.div>
        )}

        {phase === 'brand' && (
          <motion.div
            key="brand"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              position: 'relative',
              zIndex: 1,
            }}
          >
            <motion.div
              initial={{ scale: 0, rotate: -90, opacity: 0 }}
              animate={{ scale: 1, rotate: 0, opacity: 1 }}
              transition={{ type: 'spring', stiffness: 220, damping: 18 }}
              style={{
                width: 96,
                height: 96,
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                borderRadius: '26px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                boxShadow: '0 0 80px rgba(99,102,241,0.55)',
                marginBottom: '1.75rem',
              }}
            >
              <Cpu size={52} strokeWidth={1.5} />
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20, filter: 'blur(8px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              transition={{ delay: 0.25, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
              style={{
                fontSize: 'clamp(2.2rem, 7vw, 3.8rem)',
                fontWeight: 800,
                letterSpacing: '-0.04em',
                background: 'linear-gradient(120deg, #ffffff 30%, #c7d2fe 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              ML-Agent
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, letterSpacing: '0.5em' }}
              animate={{ opacity: 1, letterSpacing: '0.28em' }}
              transition={{ delay: 0.55, duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
              style={{
                marginTop: '0.85rem',
                fontSize: '0.78rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                color: 'var(--text-secondary)',
              }}
            >
              Autonomous ML Engineer
            </motion.p>

            {/* progress shimmer line */}
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 160, opacity: 1 }}
              transition={{ delay: 0.7, duration: 1.4, ease: 'easeInOut' }}
              style={{
                marginTop: '2rem',
                height: 2,
                borderRadius: 2,
                background: 'linear-gradient(90deg, transparent, var(--accent-primary), var(--accent-secondary), transparent)',
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export default IntroSequence;
