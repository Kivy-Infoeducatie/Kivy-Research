'use client';

import {
  HandTrackingProvider,
  useHandTracking
} from '@/lib/hand-tracking/hand-tracking-context';
import { HandTrackingVideo } from '@/components/playground/dev/hand-tracking-video';
import { HandCursor } from '@/components/playground/dev/hand-cursor';
import Selectable from '@/components/playground/core/selectable';

function Playground() {
  const { toggleTracking } = useHandTracking();

  return (
    <div>
      <HandTrackingVideo />
      <button onClick={toggleTracking}>Toggle</button>
      <HandCursor />
      <Selectable>
        <div className='rounded bg-white p-4 shadow-md'>
          <h2 className='text-lg font-bold'>Widget</h2>
          <p>This is a movable widget.</p>
        </div>
      </Selectable>
    </div>
  );
}

export default function () {
  return (
    <HandTrackingProvider>
      <Playground />
    </HandTrackingProvider>
  );
}
