import {
  createContext,
  ReactNode,
  RefObject,
  useContext,
  useEffect,
  useRef,
  useState
} from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
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

interface HandTrackingContextInterface {
  handTracker: HandLandmarker | null;
  modelStatus: ModelStatus;
  videoRef: RefObject<any>;
  rawLandmarks: NormalizedLandmark[][];
  landmarks: HandLandmarks[];
  handEvents: HandEvent[];

  initializeHandLandMarker(): Promise<void>;

  toggleTracking(): void;
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

enum ModelStatus {
  LOADING = 'loading',
  READY = 'ready',
  ERROR = 'error'
}

export enum HandEvent {
  PRIMARY_TOUCH,
  SECONDARY_TOUCH,
  TERTIARY_TOUCH
}

export function HandTrackingProvider({ children }: { children: ReactNode }) {
  const [modelStatus, setModelStatus] = useState<ModelStatus>(
    ModelStatus.LOADING
  );
  const [handTracker, setHandTracker] = useState<HandLandmarker | null>(null);

  const [isTracking, setIsTracking] = useState<boolean>(false);
  const [webcamRunning, setWebcamRunning] = useState<boolean>(false);

  const [rawLandmarks, setRawLandmarks] = useState<NormalizedLandmark[][]>([]);
  const [landmarks, setLandmarks] = useState<HandLandmarks[]>([]);

  const [handEvents, setHandEvents] = useState<HandEvent[]>([]);

  const videoRef = useRef(null);

  async function initializeHandLandMarker() {
    setModelStatus(ModelStatus.LOADING);

    try {
      const vision = await FilesetResolver.forVisionTasks('/wasm');

      const landMarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/models/hand_landmarker.task',
          delegate: 'GPU'
        },
        numHands: 2,
        runningMode: 'VIDEO'
      });

      setHandTracker(landMarker);
      setModelStatus(ModelStatus.READY);
    } catch (error) {
      console.error('Error initializing HandLandMarker:', error);
      setModelStatus(ModelStatus.ERROR);
    }
  }

  useEffect(() => {
    void initializeHandLandMarker();
  }, []);

  let lastVideoTime = -1;
  let animationFrameID = 0;

  const hoveredElements = useRef<Set<Element>>(new Set<Element>());
  const hoveredElementTypes = useRef<Map<Element, number>>(new Map());

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

        const events: HandEvent[] = [];

        for (let i = 0; i < parsedLandmarks.length; i++) {
          const landmark = parsedLandmarks[i];

          if (
            getDistance(landmark.index.tip, landmark.thumb.tip) <
            SECONDARY_TOUCH_DISTANCE
          ) {
            events.push(HandEvent.SECONDARY_TOUCH);

            eventPropagation(
              hoveredElements,
              hoveredElementTypes,
              landmark.index.tip.x * window.innerWidth,
              landmark.index.tip.y * window.innerHeight,
              HandEvent.SECONDARY_TOUCH,
              i
            );
          } else if (
            getDistance(landmark.index.tip, landmark.middle.tip) <
            TERTIARY_TOUCH_DISTANCE
          ) {
            events.push(HandEvent.TERTIARY_TOUCH);

            eventPropagation(
              hoveredElements,
              hoveredElementTypes,
              landmark.index.tip.x * window.innerWidth,
              landmark.index.tip.y * window.innerHeight,
              HandEvent.TERTIARY_TOUCH,
              i
            );
          } else {
            events.push(HandEvent.PRIMARY_TOUCH);

            eventPropagation(
              hoveredElements,
              hoveredElementTypes,
              landmark.index.tip.x * window.innerWidth,
              landmark.index.tip.y * window.innerHeight,
              HandEvent.PRIMARY_TOUCH,
              i
            );
          }
        }

        setHandEvents(events);
      }
    }

    if (isTracking) {
      animationFrameID = requestAnimationFrame(predictHandMarks);
    }
  }

  useEffect(() => {
    if (!videoRef.current) return;

    const constraints = {
      video: { width: 640, height: 480 }
    };

    async function enableWebcam() {
      try {
        // @ts-ignore
        videoRef.current!.srcObject =
          await navigator.mediaDevices.getUserMedia(constraints);
        // @ts-ignore
        videoRef.current!.addEventListener('loadeddata', predictHandMarks);
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    }

    if (isTracking && !webcamRunning) {
      void enableWebcam();
      setWebcamRunning(true);
    } else if (!isTracking && webcamRunning) {
      // @ts-ignore
      const tracks = videoRef.current.srcObject?.getTracks();
      tracks?.forEach((track: any) => track.stop());
      // @ts-ignore

      videoRef.current.srcObject = null;
      setWebcamRunning(false);
    }

    return () => {
      // @ts-ignore
      const tracks = videoRef.current?.srcObject?.getTracks();
      tracks?.forEach((track: any) => track.stop());
    };
  }, [isTracking, webcamRunning]);

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
        initializeHandLandMarker,
        toggleTracking,
        videoRef,
        rawLandmarks,
        landmarks,
        handEvents
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
