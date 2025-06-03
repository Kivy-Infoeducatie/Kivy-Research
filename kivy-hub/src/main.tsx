import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { KivyProvider } from './lib/contexts/kivy-provider.tsx';
import { HandTrackingProvider } from './lib/hand-tracking/hand-tracking-context.tsx';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <KivyProvider>
    <HandTrackingProvider>
      <App />
    </HandTrackingProvider>
  </KivyProvider>
);
