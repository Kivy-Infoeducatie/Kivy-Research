import { useHandTracking } from '@/lib/core/hand-tracking/hand-tracking-context';

export function StartCameraWidget() {
  const { toggleTracking, isTracking } = useHandTracking();

  if (isTracking) {
    return null;
  }

  return (
    <button
      onClick={() => {
        toggleTracking();
      }}
      className='fixed top-8 right-8 cursor-pointer rounded-full bg-white px-4 py-2 text-black'
    >
      Start Camera
    </button>
  );
}
