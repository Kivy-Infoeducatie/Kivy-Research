import './App.css';
import HomeWidget from './components/widgets/home-widget.tsx';
import './lib/fontawesome/css/fa.css';
import RecipeWidget from './components/widgets/recipe-widget.tsx';
import React, { useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const HandFingerTracking = () => {
  const videoRef = useRef(null);
  const [handLandmarker, setHandLandmarker] = useState(null);
  const [isTracking, setIsTracking] = useState(false);
  const [webcamRunning, setWebcamRunning] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);

  const hoveredElements = useRef<Set<Element>>(new Set());

  const ref = useRef<HTMLDivElement>(null);
  const ref2 = useRef<HTMLDivElement>(null);

  // MediaPipe hand landmark indices
  // Index finger tip is landmark 8
  // Thumb tip is landmark 4
  const INDEX_FINGER_TIP = 8;
  const THUMB_TIP = 4;

  // Initialize the HandLandmarker
  useEffect(() => {
    const initializeHandLandmarker = async () => {
      setIsModelLoading(true);
      try {
        // For online model:
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );

        // For local model:
        // const vision = await FilesetResolver.forVisionTasks(
        //   `${process.env.PUBLIC_URL}/models/wasm`
        // );

        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            // For local model:
            // modelAssetPath: handLandmarkerModel,
            delegate: 'GPU'
          },
          numHands: 2,
          runningMode: 'VIDEO'
        });

        setHandLandmarker(landmarker);
      } catch (error) {
        console.error('Error initializing HandLandmarker:', error);
      } finally {
        setIsModelLoading(false);
      }
    };

    initializeHandLandmarker();

    function test() {
      console.log('test');
    }

    // window.addEventListener('mousemove', test);
    //
    // return () => {
    //   window.removeEventListener('mousemove', test);
    // };
  }, []);

  // Set up webcam
  useEffect(() => {
    if (!videoRef.current) return;

    const constraints = {
      video: { width: 640, height: 480 }
    };

    const enableWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoRef.current.srcObject = stream;
        videoRef.current.addEventListener('loadeddata', predictWebcam);
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    };

    if (isTracking && !webcamRunning) {
      enableWebcam();
      setWebcamRunning(true);
    } else if (!isTracking && webcamRunning) {
      const tracks = videoRef.current.srcObject?.getTracks();
      tracks?.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setWebcamRunning(false);
    }

    return () => {
      const tracks = videoRef.current?.srcObject?.getTracks();
      tracks?.forEach((track) => track.stop());
    };
  }, [isTracking, webcamRunning]);

  let lastVideoTime = -1;
  let animationFrameId = 0;

  const predictWebcam = async () => {
    if (!handLandmarker || !videoRef.current || !isTracking) {
      return;
    }

    const video = videoRef.current;

    // Check if video has loaded and is playing
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;

      // Process the frame with HandLandmarker
      const startTimeMs = performance.now();
      const results = handLandmarker.detectForVideo(video, startTimeMs);

      // Log finger tips positions
      if (results.landmarks && results.landmarks.length > 0) {
        results.landmarks.forEach((landmarks, handIndex) => {
          const indexFingerTip = landmarks[INDEX_FINGER_TIP];
          const thumbTip = landmarks[THUMB_TIP];

          const distance = calculateDistance(indexFingerTip, thumbTip);

          // const [x, y] = transformCoordinates(
          //   1 - indexFingerTip.x,
          //   indexFingerTip.y
          // );

          const [x, y] = [
            (indexFingerTip.x * window.innerWidth + 50) * 1.30,
            (indexFingerTip.y * window.innerHeight - 150) * 1.68
          ];

          console.log(x, y);

          ref.current.style.left = `${x}px`;
          ref.current.style.top = `${y}px`;

          const elementsAtPoint = document
            .elementsFromPoint(x, y)
            .filter((el) => el !== ref.current);

          const currentElementsSet = new Set(elementsAtPoint);

          console.log(distance);

          if (distance < 0.1) {
            elementsAtPoint.forEach((element) => {
              // If this element wasn't previously hovered, trigger mouseenter
              element.dispatchEvent(
                new MouseEvent('click', {
                  clientX: x,
                  clientY: y,
                  bubbles: true,
                  cancelable: true
                })
              );
            });
          }

          // Handle mouseenter events for new elements
          elementsAtPoint.forEach((element) => {
            // If this element wasn't previously hovered, trigger mouseenter
            if (!hoveredElements.current.has(element)) {
              element.dispatchEvent(
                new MouseEvent('mousedown', {
                  clientX: x,
                  clientY: y,
                  bubbles: true,
                  cancelable: true
                })
              );
            }
          });

          hoveredElements.current.forEach((element) => {
            // If this element is no longer under the cursor, trigger mouseleave
            if (!currentElementsSet.has(element)) {
              element.dispatchEvent(
                new MouseEvent('mouseup', {
                  clientX: x,
                  clientY: y,
                  bubbles: true,
                  cancelable: true
                })
              );
            }
          });

          // Send mousemove events to all current elements
          // elementsAtPoint.forEach((element) => {
          //   element.dispatchEvent(
          //     new MouseEvent('mousemove', {
          //       clientX: x,
          //       clientY: y,
          //       bubbles: true,
          //       cancelable: true
          //     })
          //   );
          // });

          hoveredElements.current = currentElementsSet;
        });
      }
    }

    // Continue detection loop
    if (isTracking) {
      animationFrameId = requestAnimationFrame(predictWebcam);
    }
  };

  // Calculate distance between two points in 3D space
  const calculateDistance = (point1, point2) => {
    return Math.sqrt(
      Math.pow(point1.x - point2.x, 2) +
        Math.pow(point1.y - point2.y, 2) +
        Math.pow(point1.z - point2.z, 2)
    );
  };

  const toggleTracking = () => {
    console.log(23);
    if (isTracking) {
      cancelAnimationFrame(animationFrameId);
    }
    setIsTracking(!isTracking);
  };

  return (
    <main className='w-screen h-screen flex items-start'>
      <div
        ref={ref}
        className='fixed w-20 h-20 min-w-20 min-h-20 max-w-20 max-h-20 bg-orange-500'
      ></div>
      <video
        ref={videoRef}
        className='w-full max-w-lg block fixed top-10 left-10 z-[200]'
        width='640'
        height='480'
        autoPlay
        playsInline
      />
      <div
        className='w-20 h-20 block fixed top-10 left-10 z-[400] bg-amber-500'
        onClick={toggleTracking}
      />
      <div className='fixed top-4 left-4'>
        <RecipeWidget />
      </div>
      <div className='fixed bottom-4 right-4'>
        <HomeWidget />
      </div>
    </main>
  );
};

export default HandFingerTracking;
