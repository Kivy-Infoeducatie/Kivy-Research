import { createContext, ReactNode, useContext, useState } from 'react';
import { TimerStack } from '@/components/playground/widgets/timer-widget/timer-widget-types';

interface TimerWidgetContextInterface {
  stacks: TimerStack[];
  setStacks: (stacks: TimerStack[]) => void;
  addTimer: () => void;
  removeTimer: (id: string) => void;
  toggleTimer: (id: string) => void;
}

const timerWidgetContext = createContext<TimerWidgetContextInterface | null>(
  null
);

export function useTimerWidget() {
  const ctx = useContext(timerWidgetContext);

  if (!ctx) {
    throw new Error(
      'useTimerWidgetContext must be used within a TimerWidgetProvider'
    );
  }

  return ctx;
}

export function TimerWidgetProvider({ children }: { children: ReactNode }) {
  const [stacks, setStacks] = useState<TimerStack[]>([]);

  function addTimer() {}

  function removeTimer(id: string) {}

  function toggleTimer(id: string) {}

  return (
    <timerWidgetContext.Provider
      value={{
        stacks,
        setStacks,
        addTimer,
        removeTimer,
        toggleTimer
      }}
    >
      {children}
    </timerWidgetContext.Provider>
  );
}
