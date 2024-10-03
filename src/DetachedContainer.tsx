// DetachedContainer.tsx

import React, { useState, useEffect, useRef } from 'react';
import './DetachedContainer.css';
import { FaFont } from 'react-icons/fa';

interface DetachedContainerProps {
  messages: string[];
  isDarkMode: boolean;
  title: string;
  language: 'german' | 'english';
}

const fontStyles = [
  'Arial, sans-serif',
  '"Times New Roman", Times, serif',
  '"Courier New", Courier, monospace',
  '"Lucida Console", Monaco, monospace',
  '"Trebuchet MS", Helvetica, sans-serif',
];

const DetachedContainer: React.FC<DetachedContainerProps> = ({ messages, isDarkMode }) => {
  const [numMessages, setNumMessages] = useState<number>(1);
  const [textColor, setTextColor] = useState<string>(isDarkMode ? '#ffffff' : '#333333');
  const [bgColor, setBgColor] = useState<string>(isDarkMode ? '#2c2c2c' : '#f9f9f9');
  const [fontStyle, setFontStyle] = useState<string>(fontStyles[0]);
  const [fontSize, setFontSize] = useState<number>(16);

  const containerRef = useRef<HTMLDivElement>(null);

  const handleFontStyleChange = () => {
    const currentIndex = fontStyles.indexOf(fontStyle);
    const nextIndex = (currentIndex + 1) % fontStyles.length;
    setFontStyle(fontStyles[nextIndex]);
  };

  const handleFontSizeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFontSize(parseInt(event.target.value, 10));
  };

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setNumMessages(parseInt(event.target.value, 10));
  };

  const handleTextColorChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTextColor(event.target.value);
  };

  const handleBgColorChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setBgColor(event.target.value);
  };

  const uniqueMessages = messages.reduce<string[]>((acc, msg) => {
    if (acc.length === 0 || acc[acc.length - 1] !== msg) {
      acc.push(msg);
    }
    return acc;
  }, []);

  const displayedMessages = uniqueMessages.slice(-numMessages);

  useEffect(() => {
    if (isDarkMode) {
      setBgColor('#2c2c2c');
      setTextColor('#ffffff');
    } else {
      setBgColor('#f9f9f9');
      setTextColor('#333333');
    }
  }, [isDarkMode]);

  return (
    <div
      className="detached-container"
      style={{
        backgroundColor: bgColor,
        color: textColor,
        fontFamily: fontStyle,
        fontSize: `${fontSize}px`,
      }}
      ref={containerRef}
    >
      <div className="top-controls">
        <div className="controls-row" style={{display:'flex'}}>
          <div className="control-group">
            <input
              type="range"
              id="message-slider"
              name="message-slider"
              min="1"
              max="5"
              value={numMessages}
              onChange={handleSliderChange}
              className="slider"
              title="Adjust number of messages"
              aria-label="Adjust number of messages"
            />
          </div>

          <div className="control-group">
            <input
              type="range"
              id="font-size-slider"
              name="font-size-slider"
              min="12"
              max="36"
              value={fontSize}
              onChange={handleFontSizeChange}
              className="slider"
              title="Adjust font size"
              aria-label="Adjust font size"
            />
          </div>

          <div className="control-group">
            <input
              type="color"
              id="text-color-picker"
              name="text-color-picker"
              value={textColor}
              onChange={handleTextColorChange}
              className="color-picker"
              title="Select text color"
              aria-label="Select text color"
            />
          </div>

          <div className="control-group">
            <input
              type="color"
              id="bg-color-picker"
              name="bg-color-picker"
              value={bgColor}
              onChange={handleBgColorChange}
              className="color-picker"
              title="Select background color"
              aria-label="Select background color"
            />
          </div>

          <div className="control-group">
            <button
              className="font-style-button"
              onClick={handleFontStyleChange}
              title="Change Font Style"
              aria-label="Change Font Style"
            >
              <FaFont /> 
            </button>
          </div>
        </div>
      </div>
      <div className="messages-container">
        {displayedMessages.map((msg, index) => {
          const isLatest = index === displayedMessages.length - 1;
          return (
            <p key={index} className={`detached-message ${isLatest ? 'latest-message' : ''}`}>
              {msg}
            </p>
          );
        })}
      </div>
    </div>
  );
};

export default DetachedContainer;
