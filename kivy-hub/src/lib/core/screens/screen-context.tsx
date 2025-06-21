import { createContext, ReactNode, useContext, useRef } from 'react';
import { useHandTracking } from '@/lib/core/hand-tracking/hand-tracking-context';
import { HandTrackingVideo } from '@/components/playground/dev/hand-tracking-video';
import { HandCursors } from '@/components/playground/dev/hand-cursors';
import { TimerWidgetStack } from '@/components/playground/widgets/timer-widget/timer-widget-stack';
import { HomeWidget } from '@/components/playground/widgets/home-widget/home-widget';
import RecipeWidget from '@/components/playground/widgets/recipe-widget';
import { StartCameraWidget } from '@/components/playground/widgets/start-camera-widget';
import { useTimerWidget } from '@/components/playground/widgets/timer-widget/timer-widget-context';

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
  const { stacks } = useTimerWidget();

  return (
    <div>
      <HandTrackingVideo />
      <HandCursors />
      <HomeWidget />
      <StartCameraWidget />
      <RecipeWidget />
      {stacks.map((stack) => (
        <TimerWidgetStack key={stack.id} timers={stack.timers} />
      ))}
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
