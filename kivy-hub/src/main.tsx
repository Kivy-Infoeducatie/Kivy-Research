import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { KivyProvider } from './lib/contexts/kivy-provider.tsx';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <KivyProvider>
    <App />
  </KivyProvider>
);
