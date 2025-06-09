import {
  HandEvent,
  useHandTracking
} from '@/lib/hand-tracking/hand-tracking-context';
import { cn } from '@/lib/utils';

const colors = {
  [HandEvent.PRIMARY_TOUCH]: 'border-blue-500 bg-blue-500/20',
  [HandEvent.SECONDARY_TOUCH]: 'border-green-500 bg-green-500/20',
  [HandEvent.TERTIARY_TOUCH]: 'border-yellow-500 bg-yellow-500/20'
};

export function HandCursor() {
  const { landmarks, handEvents } = useHandTracking();

  return (
    <>
      {landmarks.map((landmark, index) => (
        <div
          key={index}
          className={`pointer-events-none fixed top-0 left-0 z-50 transition-transform duration-75 ease-linear`}
          style={{
            transform: `translate(${landmark.index.tip.x * 1000 - 15}px, ${landmark.index.tip.y * 1000 - 15}px)`
          }}
        >
          <div
            className={cn(
              'h-8 w-8 rounded-full border-4 shadow-md backdrop-blur-sm',
              colors[handEvents[index]]
            )}
          />
          {JSON.stringify(handEvents)}
        </div>
      ))}
    </>
  );
}
