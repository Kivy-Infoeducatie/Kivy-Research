'use client';

import { HandTrackingVideo } from '@/components/playground/dev/hand-tracking-video';
import { HandCursor } from '@/components/playground/dev/hand-cursor';
import Selectable from '@/components/playground/core/selectable';
import { useMouseSupport } from '@/lib/core/hand-tracking/use-mouse-support';
import {
  HandTrackingProvider,
  useHandTracking
} from '@/lib/core/hand-tracking/hand-tracking-context';
import { MultiProvider } from '@/components/playground/misc/multi-provider';
import { WidgetsProvider } from '@/lib/core/widgets/widget-context';

function Playground() {
  const { toggleTracking } = useHandTracking();

  return (
    <div>
      <HandTrackingVideo />
      <button onClick={toggleTracking}>Toggle</button>
      <HandCursor />
      <Selectable
        onPrimaryPress={() => {
          console.log('Primary press detected');
        }}
        onSecondaryPress={() => {
          console.log('Secondary press detected');
        }}
        onTertiaryPress={() => {
          console.log('Tertiary press detected');
        }}
      >
        <div className='rounded bg-white p-4 shadow-md'>
          <h2 className='text-lg font-bold'>Widget</h2>
          <p>This is a movable widget.</p>
        </div>
      </Selectable>
    </div>
  );
}

export default function () {
  useMouseSupport();

  return (
    <MultiProvider providers={[HandTrackingProvider, WidgetsProvider]}>
      <Playground />
    </MultiProvider>
  );
}
