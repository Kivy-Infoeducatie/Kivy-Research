import './App.css';
import './lib/fontawesome/css/fa.css';
import { useHandTracking } from './lib/hand-tracking/hand-tracking-context.tsx';
import { HandTrackingVideo } from './components/dev/hand-tracking-video.tsx';
import { HandCursor } from './components/dev/hand-cursor.tsx';
import Selectable from './components/core/selectable.tsx';
import { CalibrationOverlay } from './components/overlays/calibration-overlay.tsx';

function Widget() {
  return (
    <Selectable>
      <div className='bg-white p-4 rounded shadow-md'>
        <h2 className='text-lg font-bold'>Widget</h2>
        <p>This is a movable widget.</p>
      </div>
    </Selectable>
  );
}

export default function () {
  const { toggleTracking } = useHandTracking();

  return (
    <div>
      <HandTrackingVideo />
      <button onClick={toggleTracking}>Toggle</button>
      <HandCursor />
      <Widget />
      <CalibrationOverlay />
    </div>
  );
}
