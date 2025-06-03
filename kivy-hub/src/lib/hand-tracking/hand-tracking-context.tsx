import {
  createContext,
  MutableRefObject,
  ReactNode,
  useContext,
  useEffect,
  useRef,
  useState
} from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { NormalizedLandmark } from '@mediapipe/hands';
import { eventPropagation } from './event-propagation.ts';
import { HandLandmarks, parseLandmarksArray } from './format-landmarks.ts';

interface HandTrackingContextInterface {
  handTracker: HandLandmarker | null;
  modelStatus: ModelStatus;
  videoRef: MutableRefObject<any>;
  rawLandmarks: NormalizedLandmark[][];
  landmarks: HandLandmarks[];

  initializeHandLandMarker(): Promise<void>;

  toggleTracking(): void;
}

const handTrackingContext = createContext<HandTrackingContextInterface | null>(
  null
);

export function useHandTracking() {
  if (!handTrackingContext) {
    throw new Error(
      'useHandTracking must be used within a HandTrackingProvider'
    );
  }

  return useContext(handTrackingContext);
}

enum ModelStatus {
  LOADING = 'loading',
  READY = 'ready',
  ERROR = 'error'
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

  async function predictHandMarks() {
    if (!handTracker || !videoRef.current || !isTracking) {
      return;
    }

    const video = videoRef.current;

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;

      const startTimeMs = performance.now();
      const results = handTracker.detectForVideo(video, startTimeMs);

      if (results.landmarks && results.landmarks.length > 0) {
        setRawLandmarks(results.landmarks);

        const parsedLandmarks = parseLandmarksArray(results.landmarks);

        setLandmarks(parsedLandmarks);

        if (parsedLandmarks.length > 0) {
          eventPropagation(
            hoveredElements,
            parsedLandmarks[0].index.tip.x * 1000,
            parsedLandmarks[0].index.tip.y * 1000
          );
        }
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
        videoRef.current.srcObject =
          await navigator.mediaDevices.getUserMedia(constraints);
        videoRef.current.addEventListener('loadeddata', predictHandMarks);
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    }

    if (isTracking && !webcamRunning) {
      void enableWebcam();
      setWebcamRunning(true);
    } else if (!isTracking && webcamRunning) {
      const tracks = videoRef.current.srcObject?.getTracks();
      tracks?.forEach((track: any) => track.stop());
      videoRef.current.srcObject = null;
      setWebcamRunning(false);
    }

    return () => {
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
        landmarks
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
