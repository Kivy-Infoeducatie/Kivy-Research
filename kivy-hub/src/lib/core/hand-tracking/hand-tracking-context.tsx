import {
  createContext,
  ReactNode,
  RefObject,
  useContext,
  useRef,
  useState
} from 'react';
import { HandLandmarker } from '@mediapipe/tasks-vision';
import { NormalizedLandmark } from '@mediapipe/hands';
import {
  getDistance,
  SECONDARY_TOUCH_DISTANCE,
  TERTIARY_TOUCH_DISTANCE
} from '@/lib/math';
import {
  HandLandmarks,
  parseLandmarksArray
} from '@/lib/core/hand-tracking/format-landmarks';
import { eventPropagation } from '@/lib/core/hand-tracking/event-propagation';
import { EventRegistry } from '@/lib/core/hand-tracking/event-registry';
import { Point } from '@/lib/types';
import {
  HandEvent,
  HandTrackingEvents,
  ModelStatus
} from '@/lib/core/hand-tracking/hand-tracking-types';
import { useWebcam } from '@/lib/core/hand-tracking/use-webcam';
import { useHandLandMarker } from '@/lib/core/hand-tracking/use-hand-land-marker';

interface HandTrackingContextInterface {
  handTracker: HandLandmarker | null;
  modelStatus: ModelStatus;
  videoRef: RefObject<any>;
  rawLandmarks: NormalizedLandmark[][];
  landmarks: HandLandmarks[];
  handEvents: HandEvent[];
  eventRegistryRef: RefObject<EventRegistry<HandTrackingEvents>>;

  toggleTracking(): void;

  landmarksRef: RefObject<HandLandmarks[]>;
}

const handTrackingContext = createContext<HandTrackingContextInterface | null>(
  null
);

export function useHandTracking() {
  const ctx = useContext(handTrackingContext)!;

  if (!ctx) {
    throw new Error(
      'useHandTracking must be used within a HandTrackingProvider'
    );
  }

  return ctx;
}

export function HandTrackingProvider({ children }: { children: ReactNode }) {
  const [isTracking, setIsTracking] = useState<boolean>(false);
  const [webcamRunning, setWebcamRunning] = useState<boolean>(false);

  const [rawLandmarks, setRawLandmarks] = useState<NormalizedLandmark[][]>([]);
  const [landmarks, setLandmarks] = useState<HandLandmarks[]>([]);
  const landmarksRef = useRef<HandLandmarks[]>([]);

  const [handEvents, setHandEvents] = useState<HandEvent[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);

  const eventRegistryRef = useRef(new EventRegistry<HandTrackingEvents>());

  let lastVideoTime = -1;
  let animationFrameID = 0;

  const hoveredElements = useRef<Set<Element>>(new Set<Element>());
  const hoveredElementTypes = useRef<Map<Element, number>>(new Map());

  const { handTracker, modelStatus } = useHandLandMarker();

  async function predictHandMarks() {
    if (!handTracker || !videoRef.current || !isTracking) {
      return;
    }

    const video = videoRef.current as HTMLVideoElement;

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;

      const startTimeMs = performance.now();
      const results = handTracker.detectForVideo(video, startTimeMs);

      if (results.landmarks && results.landmarks.length > 0) {
        setRawLandmarks(results.landmarks);

        const parsedLandmarks = parseLandmarksArray(results.landmarks);

        setLandmarks(parsedLandmarks);
        landmarksRef.current = parsedLandmarks;

        const events: HandEvent[] = [];

        const eventRegistry = eventRegistryRef.current;

        function emitEvent(handEvent: HandEvent, point: Point, index: number) {
          events.push(handEvent);

          eventPropagation(
            hoveredElements,
            hoveredElementTypes,
            point.x * window.innerWidth,
            point.y * window.innerHeight,
            handEvent,
            index
          );

          eventRegistry.emit(
            'touch-move',
            {
              x: point.x * window.innerWidth,
              y: point.y * window.innerHeight
            },
            index,
            handEvent
          );
        }

        for (let i = 0; i < parsedLandmarks.length; i++) {
          const landmark = parsedLandmarks[i];

          if (
            getDistance(landmark.index.tip, landmark.thumb.tip) <
            SECONDARY_TOUCH_DISTANCE
          ) {
            emitEvent(HandEvent.SECONDARY_TOUCH, landmark.index.tip, i);
          } else if (
            getDistance(landmark.index.tip, landmark.middle.tip) <
            TERTIARY_TOUCH_DISTANCE
          ) {
            emitEvent(HandEvent.TERTIARY_TOUCH, landmark.index.tip, i);
          } else {
            emitEvent(HandEvent.PRIMARY_TOUCH, landmark.index.tip, i);
          }
        }

        setHandEvents(events);
      }
    }

    if (isTracking) {
      animationFrameID = requestAnimationFrame(predictHandMarks);
    }
  }

  useWebcam(
    videoRef,
    isTracking,
    webcamRunning,
    setWebcamRunning,
    predictHandMarks
  );

  function toggleTracking() {
    if (isTracking) {
      cancelAnimationFrame(animationFrameID);
    }
    setIsTracking(!isTracking);
  }

  return (
    <handTrackingContext.Provider
      value={{
        handTracker,
        modelStatus,
        toggleTracking,
        landmarksRef,
        videoRef,
        rawLandmarks,
        landmarks,
        handEvents,
        eventRegistryRef
      }}
    >
      {children}
      <video
        ref={videoRef}
        className='hidden'
        width='640'
        height='480'
        autoPlay
        playsInline
      />
    </handTrackingContext.Provider>
  );
}
