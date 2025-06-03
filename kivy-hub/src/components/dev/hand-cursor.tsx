import { useHandTracking } from '../../lib/hand-tracking/hand-tracking-context.tsx';

export function HandCursor() {
  const { landmarks } = useHandTracking();

  return (
    <>
      {landmarks.map((landmark, index) => (
        <div
          key={index}
          className={`fixed top-0 left-0 pointer-events-none transition-transform duration-75 ease-linear z-50 `}
          style={{
            transform: `translate(${landmark.index.tip.x * 1000 - 15}px, ${landmark.index.tip.y * 1000 - 15}px)`
          }}
        >
          <div className='w-8 h-8 rounded-full border-4 border-blue-500 bg-blue-500/20 backdrop-blur-sm shadow-md' />
        </div>
      ))}
    </>
  );
}
