// src/index.tsx

import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import DetachedContainer from './DetachedContainer';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

const container = document.getElementById('root');
const root = createRoot(container!);
root.render(
  <Router>
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/detached" element={<DetachedContainerWrapper />} />
    </Routes>
  </Router>
);

// Wrapper to extract query parameters and pass props
function DetachedContainerWrapper() {
  const params = new URLSearchParams(window.location.search);
  const language = params.get('language') as 'german' | 'english';

  return (
    <DetachedContainer
      messages={[]} // Initial messages; will be updated via IPC
      isDarkMode={true} // Initial theme; will be updated via IPC
      title={language === 'german' ? 'German Transcription' : 'English Translation'}
      language={language}
    />
  );
}
