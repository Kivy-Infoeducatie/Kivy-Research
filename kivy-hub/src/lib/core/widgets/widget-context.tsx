import { createContext, ReactNode, useContext } from 'react';

interface WidgetsContext {}

const widgetsContext = createContext<WidgetsContext | null>(null);

export function useWidgets() {
  const ctx = useContext(widgetsContext);

  if (!ctx) {
    throw new Error('useWidgets must be used within a WidgetsContextProvider');
  }

  return ctx;
}

export function WidgetsProvider({ children }: { children: ReactNode }) {
  return (
    <widgetsContext.Provider value={{}}>{children}</widgetsContext.Provider>
  );
}
