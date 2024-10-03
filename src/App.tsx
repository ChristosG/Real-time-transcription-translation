import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import './App.css';
import DetachedContainer from './DetachedContainer';
import { FaExternalLinkAlt } from 'react-icons/fa';

interface TooltipProps {
  visible: boolean;
  message: string;
  position: React.CSSProperties;
}

const Tooltip: React.FC<TooltipProps> = ({ visible, message, position }) => {
  if (!visible) return null;

  return (
    <div className="tooltip" style={position}>
      <span className="tooltip-text">{message}</span>
    </div>
  );
};

function App() {
  const [isDarkMode, setIsDarkMode] = useState<boolean>(true);

  const [connectionTooltipVisible, setConnectionTooltipVisible] = useState<boolean>(false);
  const [themeTooltipVisible, setThemeTooltipVisible] = useState<boolean>(false);
  const [betterTranslationTooltipVisible, setBetterTranslationTooltipVisible] = useState<boolean>(false);

  const [currentLineGerman, setCurrentLineGerman] = useState<string>('');
  const [previousLinesGerman, setPreviousLinesGerman] = useState<string[]>([]);

  const [currentLineEnglish, setCurrentLineEnglish] = useState<string>('');
  const [previousLinesEnglish, setPreviousLinesEnglish] = useState<string[]>([]);

  const [useBetterTranslation, setUseBetterTranslation] = useState<boolean>(false);
  const [currentBetterTranslation, setCurrentBetterTranslation] = useState<string>('');
  const [finalizedBetterTranslations, setFinalizedBetterTranslations] = useState<string[]>([]);

  const scrollContainerGermanRef = useRef<HTMLDivElement>(null);
  const scrollContainerEnglishRef = useRef<HTMLDivElement>(null);

  const [isAutoScrollGerman, setIsAutoScrollGerman] = useState<boolean>(true);
  const [isAutoScrollEnglish, setIsAutoScrollEnglish] = useState<boolean>(true);

  const [newMessagesGerman, setNewMessagesGerman] = useState<boolean>(false);
  const [newMessagesEnglish, setNewMessagesEnglish] = useState<boolean>(false);

  const autoScrollTimerGerman = useRef<NodeJS.Timeout | null>(null);
  const autoScrollTimerEnglish = useRef<NodeJS.Timeout | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const retryTimeoutDuration = 3000;

  const [detachedGermanWindow, setDetachedGermanWindow] = useState<Window | null>(null);
  const [detachedEnglishWindow, setDetachedEnglishWindow] = useState<Window | null>(null);

  const germanContainerRef = useRef<HTMLDivElement | null>(null);
  const englishContainerRef = useRef<HTMLDivElement | null>(null);
  const germanRootRef = useRef<ReactDOM.Root | null>(null);
  const englishRootRef = useRef<ReactDOM.Root | null>(null);

  const toggleTheme = () => {
    setIsDarkMode((prevMode) => !prevMode);
  };

  const exportTranscription = (lines: string[], filename: string) => {
    const element = document.createElement('a');
    const file = new Blob([lines.join('\n')], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = filename;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const initializeWebSocket = () => {
    // const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    // const wsPort = '7000';
    // const ws = new WebSocket(`${wsProtocol}://${window.location.hostname}:${wsPort}/ws`);
    
    // const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    // const ws = new WebSocket(`${wsProtocol}://${window.location.hostname}/ws`);
    
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    let wsPort = '';
    
    // Use port 7000 only when running on localhost (development)
    if (window.location.hostname === 'localhost') {
      wsPort = ':7000';
    }
    
    const ws = new WebSocket(`${wsProtocol}://${window.location.hostname}${wsPort}/ws`);
    


    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        const channel = message.channel;
        const data = typeof message.data === 'string' ? JSON.parse(message.data) : message.data;

        if (channel === 'transcriptions') {
          if (data.transcription) {
            const germanText = data.transcription;
            setCurrentLineGerman(germanText);
            setPreviousLinesGerman((prevLines) => [...prevLines, germanText]);

            if (!isAutoScrollGerman) {
              setNewMessagesGerman(true);
            }
          }
        } else if (channel === 'translations') {
          if (data.translation) {
            const englishText = data.translation;
            setCurrentLineEnglish(englishText);
            setPreviousLinesEnglish((prevLines) => [...prevLines, englishText]);

            if (!isAutoScrollEnglish) {
              setNewMessagesEnglish(true);
            }
          }
        } else if (channel === 'better_translations') {
          if (data.translation) {
            const betterText = data.translation;

            if (data.finalized) {
              setFinalizedBetterTranslations((prev) => [...prev, betterText]);
              setCurrentBetterTranslation('');
            } else {
              setCurrentBetterTranslation(betterText);
            }

            if (!isAutoScrollEnglish) {
              setNewMessagesEnglish(true);
            }
          }
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected, attempting to reconnect...');
      setIsConnected(false);
      scheduleReconnect();
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  };

  const scheduleReconnect = () => {
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
    }
    retryTimeoutRef.current = setTimeout(() => {
      console.log('Reconnecting WebSocket...');
      initializeWebSocket();
    }, retryTimeoutDuration);
  };

  useEffect(() => {
    initializeWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
      if (detachedGermanWindow) detachedGermanWindow.close();
      if (detachedEnglishWindow) detachedEnglishWindow.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-scroll effects for German
  useEffect(() => {
    if (isAutoScrollGerman && scrollContainerGermanRef.current) {
      scrollContainerGermanRef.current.scrollTop = scrollContainerGermanRef.current.scrollHeight;
      setNewMessagesGerman(false);
    }
  }, [previousLinesGerman, isAutoScrollGerman, currentLineGerman]);

  // Auto-scroll effects for English
  useEffect(() => {
    if (isAutoScrollEnglish && scrollContainerEnglishRef.current) {
      scrollContainerEnglishRef.current.scrollTop = scrollContainerEnglishRef.current.scrollHeight;
      setNewMessagesEnglish(false);
    }
  }, [
    previousLinesEnglish,
    finalizedBetterTranslations,
    currentBetterTranslation,
    isAutoScrollEnglish,
    currentLineEnglish,
  ]);

  // Scroll handlers
  const handleScrollGerman = useCallback(
    (event: React.UIEvent<HTMLDivElement, UIEvent>) => {
      const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 20;

      if (isAtBottom) {
        setIsAutoScrollGerman(true);
        setNewMessagesGerman(false);
      } else {
        setIsAutoScrollGerman(false);
      }

      if (!isAutoScrollGerman) {
        if (autoScrollTimerGerman.current) {
          clearTimeout(autoScrollTimerGerman.current);
        }

        autoScrollTimerGerman.current = setTimeout(() => {
          setIsAutoScrollGerman(true);
        }, 15000);
      }
    },
    [isAutoScrollGerman]
  );

  const handleScrollEnglish = useCallback(
    (event: React.UIEvent<HTMLDivElement, UIEvent>) => {
      const { scrollTop, scrollHeight, clientHeight } = event.currentTarget;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 20;

      if (isAtBottom) {
        setIsAutoScrollEnglish(true);
        setNewMessagesEnglish(false);
      } else {
        setIsAutoScrollEnglish(false);
      }

      if (!isAutoScrollEnglish) {
        if (autoScrollTimerEnglish.current) {
          clearTimeout(autoScrollTimerEnglish.current);
        }

        autoScrollTimerEnglish.current = setTimeout(() => {
          setIsAutoScrollEnglish(true);
        }, 15000);
      }
    },
    [isAutoScrollEnglish]
  );

  const [isMobile, setIsMobile] = useState<boolean>(false);

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      setIsMobile(width < 768);
    };

    handleResize();

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Active tab state for mobile
  const [activeTab, setActiveTab] = useState<'german' | 'english'>('german');

  // detach transcription windows
  const detachWindow = (
    language: 'german' | 'english',
    messages: string[],
    detachedWindowState: Window | null,
    setDetachedWindowState: React.Dispatch<React.SetStateAction<Window | null>>,
    containerRef: React.RefObject<HTMLDivElement>,
    rootRef: React.MutableRefObject<ReactDOM.Root | null>
  ) => {
    if (detachedWindowState && !detachedWindowState.closed) {
      detachedWindowState.focus();
      return;
    }

    const newWindow = window.open('', '', 'width=600,height=220,left=100,top=300,scrollbars=yes');

    if (newWindow) {
      newWindow.document.title = `Detached ${language === 'german' ? 'German Transcription' : 'English Translation'}`;
      newWindow.document.body.style.margin = '0';
      newWindow.document.body.style.padding = '0';
      newWindow.document.body.style.backgroundColor = isDarkMode ? '#2c2c2c' : '#f9f9f9';
      newWindow.document.body.style.color = isDarkMode ? '#ffffff' : '#333333';

      const container = newWindow.document.createElement('div');
      newWindow.document.body.appendChild(container);
      language === 'german' ? (germanContainerRef.current = container) : (englishContainerRef.current = container);

      const root = ReactDOM.createRoot(container);
      language === 'german' ? (germanRootRef.current = root) : (englishRootRef.current = root);
      root.render(
        <DetachedContainer
          messages={messages}
          isDarkMode={isDarkMode}
          title={language === 'german' ? 'German Transcription' : 'English Translation'}
          language={language}
        />
      );

      setDetachedWindowState(newWindow);

      newWindow.onbeforeunload = () => {
        setDetachedWindowState(null);
        rootRef.current = null;
        language === 'german' ? (germanContainerRef.current = null) : (englishContainerRef.current = null);
      };
    }
  };

  const germanMessages = [...previousLinesGerman, currentLineGerman].filter(Boolean);
  const englishMessages = useBetterTranslation
    ? [...finalizedBetterTranslations, currentBetterTranslation].filter(Boolean)
    : [...previousLinesEnglish, currentLineEnglish].filter(Boolean);

  // update detached windows when messages change
  useEffect(() => {
    if (germanRootRef.current && germanContainerRef.current) {
      if (detachedGermanWindow && !detachedGermanWindow.closed) {
        germanRootRef.current.render(
          <DetachedContainer
            messages={germanMessages}
            isDarkMode={isDarkMode}
            title="German Transcription"
            language="german"
          />
        );
      } else {
        // clean up if the window is closed
        germanRootRef.current.unmount();
        germanRootRef.current = null;
        germanContainerRef.current = null;
        setDetachedGermanWindow(null);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [germanMessages, isDarkMode]);

  useEffect(() => {
    if (englishRootRef.current && englishContainerRef.current) {
      if (detachedEnglishWindow && !detachedEnglishWindow.closed) {
        englishRootRef.current.render(
          <DetachedContainer
            messages={englishMessages}
            isDarkMode={isDarkMode}
            title="English Translation"
            language="english"
          />
        );
      } else {
        // clean up if the window is closed
        englishRootRef.current.unmount();
        englishRootRef.current = null;
        englishContainerRef.current = null;
        setDetachedEnglishWindow(null);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [englishMessages, isDarkMode]);

  return (
    <div className={`container ${isDarkMode ? 'dark-container' : 'light-container'}`}>
      <header className={`header ${isDarkMode ? 'dark-header' : 'light-header'}`}>
        <div className="header-title-container">
          <h1 className={`header-title ${isDarkMode ? 'dark-text-title' : 'light-text-title'}`}>
            Transcribe & Translate
          </h1>
          <div
            className="connection-container"
            onMouseEnter={() => setConnectionTooltipVisible(true)}
            onMouseLeave={() => setConnectionTooltipVisible(false)}
            onTouchStart={() => setConnectionTooltipVisible(true)}
            onTouchEnd={() => setConnectionTooltipVisible(false)}
          >
            <div className={`connection-indicator ${isConnected ? 'connected' : 'disconnected'}`} />
            <Tooltip
              visible={connectionTooltipVisible}
              message={isConnected ? 'Connected' : 'Disconnected'}
              position={{ top: '30px', left: '10px' }}
            />
          </div>
        </div>

        <div className="theme-toggle">
          {!isMobile && (
            <span className={`theme-text ${isDarkMode ? 'dark-text' : 'light-text'}`}>
              {isDarkMode ? 'Dark Mode' : 'Light Mode'}
            </span>
          )}
          {isMobile && (
            <div
              className="theme-toggle-mobile"
              onMouseEnter={() => setThemeTooltipVisible(true)}
              onMouseLeave={() => setThemeTooltipVisible(false)}
              onTouchStart={() => setThemeTooltipVisible(true)}
              onTouchEnd={() => setThemeTooltipVisible(false)}
            >
              <label className="switch">
                <input type="checkbox" checked={isDarkMode} onChange={toggleTheme} />
                <span className="slider round"></span>
              </label>
              <Tooltip
                visible={themeTooltipVisible && isMobile}
                message={isDarkMode ? 'Dark Mode' : 'Light Mode'}
                position={{ top: '30px', left: '10px' }}
              />
            </div>
          )}
          {!isMobile && (
            <label className="switch">
              <input type="checkbox" checked={isDarkMode} onChange={toggleTheme} />
              <span className="slider round"></span>
            </label>
          )}
        </div>
      </header>

      {isMobile && (
        <div className="tabs-container">
          <button
            className={`tab-button ${activeTab === 'german' ? 'active-tab-button' : ''}`}
            onClick={() => setActiveTab('german')}
          >
            German
          </button>
          <button
            className={`tab-button ${activeTab === 'english' ? 'active-tab-button' : ''}`}
            onClick={() => setActiveTab('english')}
          >
            English
          </button>
        </div>
      )}

      <main className="main-content">
        {(!isMobile || activeTab === 'german') && (
          <section className={`section ${isDarkMode ? 'dark-section' : 'light-section'}`}>
            <div className="section-header">
              <h2 className={`section-title ${isDarkMode ? 'dark-text-title' : 'light-text-title'}`}>
                German Transcription
              </h2>
              <div className="section-actions">
                <button
                  onClick={() =>
                    exportTranscription(germanMessages, 'German_Transcription.txt')
                  }
                  className="export-button"
                >
                  Export
                </button>
              </div>
            </div>
            <div
              ref={scrollContainerGermanRef}
              className={`history-container ${isDarkMode ? 'dark-scroll-view' : 'light-scroll-view'}`}
              onScroll={handleScrollGerman}
            >
              {previousLinesGerman.length === 0 && !currentLineGerman && (
                <p className={`placeholder-text ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                  No transcriptions yet. Speak to start...
                </p>
              )}
              {previousLinesGerman.map((line, index) => (
                <p key={index} className={`previous-line ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                  {line}
                </p>
              ))}
            </div>
            <div
              className={`current-container ${isDarkMode ? 'dark-current-container' : 'light-current-container'}`}
            >
              <p className={`current-line ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                {currentLineGerman || 'Listening...'}
              </p>
              {/* Detach Button */}
              {!isMobile && (
                <button
                  className="detach-button"
                  onClick={() => {
                    detachWindow(
                      'german',
                      germanMessages,
                      detachedGermanWindow,
                      setDetachedGermanWindow,
                      germanContainerRef,
                      germanRootRef
                    );
                  }}
                  title="Detach German Transcription"
                >
                  <FaExternalLinkAlt />
                </button>
              )}
            </div>
            {newMessagesGerman && isMobile && (
              <button
                className="new-message-button"
                onClick={() => {
                  if (scrollContainerGermanRef.current) {
                    scrollContainerGermanRef.current.scrollTop =
                      scrollContainerGermanRef.current.scrollHeight;
                  }
                  setNewMessagesGerman(false);
                }}
              >
                New Messages
              </button>
            )}
          </section>
        )}

        {(!isMobile || activeTab === 'english') && (
          <section className={`section ${isDarkMode ? 'dark-section' : 'light-section'}`}>
            <div className="section-header">
              <h2 className={`section-title ${isDarkMode ? 'dark-text-title' : 'light-text-title'}`} 
                  style={{display:'flex', flexDirection:'row'}}>
                English Translation

                <div
                  className="better-translation-toggle" style={{marginLeft:'10px'}}
                  onMouseEnter={() => setBetterTranslationTooltipVisible(true)}
                  onMouseLeave={() => setBetterTranslationTooltipVisible(false)}
                  onTouchStart={() => setBetterTranslationTooltipVisible(true)}
                  onTouchEnd={() => setBetterTranslationTooltipVisible(false)}
                >
                  <label className="switch">
                    <input
                      type="checkbox"
                      checked={useBetterTranslation}
                      onChange={() => setUseBetterTranslation(!useBetterTranslation)}
                    />
                    <span className="slider round"></span>
                  </label>
                  <Tooltip
                    visible={betterTranslationTooltipVisible}
                    message="Just better"
                    position={{ top: '14px', left: '300px' }}
                  />
                </div>
              </h2>
              
              <div className="section-actions">
                <button
                  onClick={() =>
                    exportTranscription(englishMessages, 'English_Translation.txt')
                  }
                  className="export-button"
                >
                  Export
                </button>
                
              </div>
            </div>
            <div
              ref={scrollContainerEnglishRef}
              className={`history-container ${isDarkMode ? 'dark-scroll-view' : 'light-scroll-view'}`}
              onScroll={handleScrollEnglish}
            >
              {useBetterTranslation ? (
                <>
                  {finalizedBetterTranslations.length === 0 && !currentBetterTranslation && (
                    <p className={`placeholder-text ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                      No better translations yet.
                    </p>
                  )}
                  {finalizedBetterTranslations.map((line, index) => (
                    <p key={index} className={`previous-line ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                      {line}
                    </p>
                  ))}
                </>
              ) : (
                <>
                  {previousLinesEnglish.length === 0 && !currentLineEnglish && (
                    <p className={`placeholder-text ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                      No translations yet.
                    </p>
                  )}
                  {previousLinesEnglish.map((line, index) => (
                    <p key={index} className={`previous-line ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                      {line}
                    </p>
                  ))}
                </>
              )}
            </div>
            <div
              className={`current-container ${isDarkMode ? 'dark-current-container' : 'light-current-container'}`}
            >
              <p className={`current-line ${isDarkMode ? 'dark-text' : 'light-text'}`}>
                {useBetterTranslation
                  ? currentBetterTranslation || 'Awaiting better translation...'
                  : currentLineEnglish || 'Awaiting translation...'}
              </p>
              {!isMobile && (
                <button
                  className="detach-button"
                  onClick={() => {
                    detachWindow(
                      'english',
                      englishMessages,
                      detachedEnglishWindow,
                      setDetachedEnglishWindow,
                      englishContainerRef,
                      englishRootRef
                    );
                  }}
                  title="Detach English Translation"
                >
                  <FaExternalLinkAlt />
                </button>
              )}
            </div>
            {newMessagesEnglish && isMobile && (
              <button
                className="new-message-button"
                onClick={() => {
                  if (scrollContainerEnglishRef.current) {
                    scrollContainerEnglishRef.current.scrollTop =
                      scrollContainerEnglishRef.current.scrollHeight;
                  }
                  setNewMessagesEnglish(false);
                }}
              >
                New Messages
              </button>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
