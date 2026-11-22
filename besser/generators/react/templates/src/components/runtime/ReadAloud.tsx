import React, { useEffect, useRef, useState } from 'react';

export const ReadAloud: React.FC<{ src: string; summary?: boolean; children: React.ReactNode }> = ({ src, summary, children }) => {
  const audio = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState(false);
  useEffect(() => {
    const stopOther = (event: Event) => {
      if (event.target !== audio.current) audio.current?.pause();
    };
    document.addEventListener('play', stopOther, true);
    return () => { document.removeEventListener('play', stopOther, true); audio.current?.pause(); };
  }, []);
  return <div>
    {children}
    <div role="group" aria-label={summary ? 'Listen to summary' : 'Listen to full text'} style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, margin: '8px 0 16px' }}>
      <button type="button" aria-label={playing ? 'Pause' : summary ? 'Listen to summary' : 'Listen to full text'} aria-pressed={playing} style={{ padding: '6px 12px', minHeight: 34, border: '1px solid #1d4ed8', borderRadius: 6, background: '#fff', color: '#1d4ed8', font: '600 1em/1.2 system-ui,sans-serif', cursor: 'pointer' }} onClick={() => {
        const element = audio.current;
        if (!element) return;
        if (element.paused) { setError(false); void element.play().catch(() => setError(true)); }
        else element.pause();
      }}>{playing ? 'Ⅱ Pause' : summary ? '▶ Play summary' : '▶ Play'}</button>
      <button type="button" aria-label="Stop" style={{ padding: '6px 12px', minHeight: 34, border: '1px solid #cbd5e1', borderRadius: 6, background: '#fff', color: '#334155', font: '600 1em/1.2 system-ui,sans-serif', cursor: 'pointer' }} onClick={() => { if (audio.current) { audio.current.pause(); audio.current.currentTime = 0; } }}>■ Stop</button>
      <audio ref={audio} src={src} preload="none" hidden onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} onError={() => setError(true)} />
      <small style={{ color: '#475569', font: '0.85em/1.4 system-ui,sans-serif' }}>AI-generated voice</small>
      {error && <span role="status">Audio could not be played.</span>}
    </div>
  </div>;
};
