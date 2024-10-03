// src/global.d.ts

export {};

declare global {
  interface Window {
    electronAPI: {
      createDetachedWindow: (data: {
        title: string;
        language: 'german' | 'english';
        messages: string[];
        isDarkMode: boolean;
      }) => void;
      onInitialData: (callback: (data: { messages: string[]; isDarkMode: boolean }) => void) => void;
      toggleTheme: (data: { isDarkMode: boolean }) => void;
      onThemeChanged: (callback: (data: { isDarkMode: boolean }) => void) => void;
    };
  }
}
