import { createContext, ReactNode, useContext, useRef } from 'react';
import { useHandTracking } from '@/lib/core/hand-tracking/hand-tracking-context';
import { HandTrackingVideo } from '@/components/playground/dev/hand-tracking-video';
import { HandCursor } from '@/components/playground/dev/hand-cursor';
import { useWidgets } from '@/lib/core/widgets/widget-context';
import { Selectable } from '@/components/playground/core/selectable';
import { TimerWidgetStack } from '@/components/playground/widgets/timer-widget/timer-widget-stack';

interface ScreenContextInterface {}

const screenContext = createContext<ScreenContextInterface | null>(null);

export function useScreenContext() {
  const ctx = useContext(screenContext);

  if (!ctx) {
    throw new Error('useScreenContext must be used within a ScreenProvider');
  }

  return ctx;
}

interface Screen {
  id: string;
  Component: () => ReactNode;
  widgets: ReactNode[];
}

function Playground() {
  const { toggleTracking } = useHandTracking();

  const { addTimer, timers } = useWidgets();

  return (
    <div>
      {/*<HandTrackingVideo />*/}
      <button className='pl-[700px]' onClick={toggleTracking}>
        Toggle
      </button>
      <HandCursor />
      {timers.map((timer) => (
        <Selectable onPrimaryPress={() => {}} key={timer.id}>
          {timer.time}
        </Selectable>
      ))}
      <button
        onClick={() => {
          addTimer(40);
        }}
      >
        Add timer
      </button>
      <TimerWidgetStack
        timers={[
          {
            id: 'a',
            title: 'Timer 1',
            currentTime: 20,
            totalTime: 40,
            isRunning: true
          },
          {
            id: 'b',
            title: 'Timer 2',
            currentTime: 20,
            totalTime: 40,
            isRunning: true
          },
          {
            id: 'c',
            title: 'Timer 3',
            currentTime: 20,
            totalTime: 40,
            isRunning: true
          }
        ]}
      />
    </div>
  );
}

export function ScreenProvider() {
  const screensRef = useRef<Screen[]>([]);

  screensRef.current = [
    {
      id: 'default',
      Component: Playground,
      widgets: []
    }
  ];

  return (
    <screenContext.Provider value={{}}>
      {screensRef.current.map((screen) => (
        <screen.Component key={screen.id} />
      ))}
    </screenContext.Provider>
  );
}
