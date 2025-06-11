import { createContext, ReactNode, useContext, useState } from 'react';

interface WidgetsContext {
  addTimer: (time: number) => void;
  timers: {
    time: number;
    x: number;
    y: number;
    id: string;
  }[];
}

const widgetsContext = createContext<WidgetsContext | null>(null);

export function useWidgets() {
  const ctx = useContext(widgetsContext);

  if (!ctx) {
    throw new Error('useWidgets must be used within a WidgetsContextProvider');
  }

  return ctx;
}

export function WidgetsProvider({ children }: { children: ReactNode }) {
  const [timers, setTimers] = useState<
    {
      time: number;
      x: number;
      y: number;
      id: string;
    }[]
  >([]);

  function addTimer(time: number) {
    setTimers((prev) => [
      ...prev,
      {
        time,
        x: 0,
        y: 0,
        id: Math.random().toString(36).substring(2, 15)
      }
    ]);
  }

  return (
    <widgetsContext.Provider
      value={{
        addTimer,
        timers
      }}
    >
      {children}
    </widgetsContext.Provider>
  );
}
