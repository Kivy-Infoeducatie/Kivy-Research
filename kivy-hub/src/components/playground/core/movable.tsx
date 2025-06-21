import {
  Selectable,
  SelectableProps
} from '@/components/playground/core/selectable';
import { Point } from '@/lib/types';
import { useHandTracking } from '@/lib/core/hand-tracking/hand-tracking-context';
import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';

type MovableProps = SelectableProps & {
  initialPos?: Point;
};

export function Movable({
  className,
  style,
  initialPos,
  ...props
}: MovableProps) {
  const { eventRegistryRef, landmarksRef, mousePositionRef } =
    useHandTracking();

  const [isDragging, setIsDragging] = useState<boolean>(false);

  const positionRef = useRef<Point>(initialPos ?? { x: 0, y: 0 });

  const movementRef = useRef<Point>({
    x: 0,
    y: 0
  });

  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isDragging) return;

    function onTouchMove(position: Point) {
      positionRef.current = {
        x: position.x - movementRef.current.x,
        y: position.y - movementRef.current.y
      };

      elementRef.current!.style.translate = `${positionRef.current.x}px ${positionRef.current.y}px`;
    }

    const eventRegistry = eventRegistryRef.current;

    eventRegistry.on('touch-move', onTouchMove);

    return () => {
      eventRegistry.off('touch-move', onTouchMove);
    };
  }, [isDragging]);

  return (
    <Selectable
      ref={elementRef}
      className={cn('!fixed top-0 left-0', className)}
      style={{
        translate: `${initialPos?.x ?? 0}px ${initialPos?.y ?? 0}px`,
        ...style
      }}
      onSecondaryPress={(e) => {
        setIsDragging(true);

        if (landmarksRef.current.length > 0) {
          movementRef.current.x =
            landmarksRef.current[0].index.tip.x * window.innerWidth -
            positionRef.current.x;
          movementRef.current.y =
            landmarksRef.current[0].index.tip.y * window.innerHeight -
            positionRef.current.y;
        } else {
          movementRef.current.x =
            mousePositionRef.current.x - positionRef.current.x;
          movementRef.current.y =
            mousePositionRef.current.y - positionRef.current.y;
        }
      }}
      onSecondaryRelease={() => {
        setIsDragging(false);
      }}
      {...props}
    />
  );
}
