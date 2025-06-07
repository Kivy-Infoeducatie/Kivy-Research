import { useHandTracking } from '../../lib/hand-tracking/hand-tracking-context.tsx';
import { useEffect, useRef } from 'react';
import { drawLandmarks } from '../../lib/hand-tracking/draw-landmarks.ts';

export function HandTrackingVideo() {
  const { modelStatus, rawLandmarks, toggleTracking, videoRef } =
    useHandTracking();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();

  function draw() {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);

    ctx.drawImage(videoRef.current, 0, 0, w, h);

    for (const landmark of rawLandmarks) {
      drawLandmarks(ctx, landmark, w, h);
    }

    ctx.restore();
  }

  function renderLoop() {
    draw();
    animationFrameRef.current = requestAnimationFrame(renderLoop);
  }

  useEffect(() => {
    animationFrameRef.current = requestAnimationFrame(renderLoop);
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [rawLandmarks]);

  return (
    <div className='rounded-full'>
      <canvas ref={canvasRef} width={640} height={480} />
    </div>
  );
}
