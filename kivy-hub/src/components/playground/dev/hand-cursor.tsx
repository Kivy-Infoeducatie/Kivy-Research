import { useHandTracking } from '@/lib/hand-tracking/hand-tracking-context';

export function HandCursor() {
  const { landmarks } = useHandTracking();

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
          <div className='h-8 w-8 rounded-full border-4 border-blue-500 bg-blue-500/20 shadow-md backdrop-blur-sm' />
        </div>
      ))}
    </>
  );
}
