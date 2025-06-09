import { HTMLAttributes, ReactNode, useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';

type HoldableProps = HTMLAttributes<HTMLDivElement> & {
  children: ReactNode;
  onPrimaryPress?: () => void;
  onSecondaryPress?: () => void;
  onTertiaryPress?: () => void;
  delay?: number;
  forceSelect?: boolean;
  stopPropagation?: boolean;
  showFeedback?: boolean;
};

const colors = {
  0: 'blue',
  1: 'green',
  2: 'yellow'
};

export default function Selectable({
  children,
  onPrimaryPress,
  onSecondaryPress,
  onTertiaryPress,
  delay = 500,
  stopPropagation = false,
  showFeedback = true,
  ...props
}: HoldableProps) {
  const [selected, setSelected] = useState<number>(-1);
  const [selecting, setSelecting] = useState(false);
  const selectingRef = useRef(false);
  const timeOutRef = useRef<NodeJS.Timeout>(null);
  const divRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = divRef.current;
    if (!el) return;

    // Create stable handler references
    const onPrimaryDown = (e: Event) => {
      if (stopPropagation) e.stopPropagation();
      setSelecting(true);
      selectingRef.current = true;
      timeOutRef.current = setTimeout(() => {
        if (selectingRef.current) {
          setSelected(0);
          setSelecting(false);
          onPrimaryPress?.();
        }
      }, delay);
    };

    const onPrimaryUp = (e: Event) => {
      if (stopPropagation) e.stopPropagation();
      console.log('cleared');
      if (timeOutRef.current) clearTimeout(timeOutRef.current);
      selectingRef.current = false;
      setSelecting(false);
      setSelected(-1);
    };

    const onSecondaryDown = (e: Event) => {
      if (stopPropagation) e.stopPropagation();
      setSelecting(true);
      selectingRef.current = true;
      timeOutRef.current = setTimeout(() => {
        if (selectingRef.current) {
          setSelected(1);
          setSelecting(false);
          onSecondaryPress?.();
        }
      }, delay);
    };

    const onSecondaryUp = (e: Event) => {
      if (stopPropagation) e.stopPropagation();
      if (timeOutRef.current) clearTimeout(timeOutRef.current);
      selectingRef.current = false;
      setSelecting(false);
      setSelected(-1);
    };

    const onTertiaryDown = (e: Event) => {
      if (stopPropagation) e.stopPropagation();
      setSelecting(true);
      selectingRef.current = true;
      timeOutRef.current = setTimeout(() => {
        if (selectingRef.current) {
          setSelected(2);
          setSelecting(false);
          onTertiaryPress?.();
        }
      }, delay);
    };

    const onTertiaryUp = (e: Event) => {
      if (stopPropagation) e.stopPropagation();
      if (timeOutRef.current) clearTimeout(timeOutRef.current);
      selectingRef.current = false;
      setSelecting(false);
      setSelected(-1);
    };

    el.addEventListener('primary-touch-down', onPrimaryDown);
    el.addEventListener('primary-touch-up', onPrimaryUp);
    el.addEventListener('secondary-touch-down', onSecondaryDown);
    el.addEventListener('secondary-touch-up', onSecondaryUp);
    el.addEventListener('tertiary-touch-down', onTertiaryDown);
    el.addEventListener('tertiary-touch-up', onTertiaryUp);

    return () => {
      el.removeEventListener('primary-touch-down', onPrimaryDown);
      el.removeEventListener('primary-touch-up', onPrimaryUp);
      el.removeEventListener('secondary-touch-down', onSecondaryDown);
      el.removeEventListener('secondary-touch-up', onSecondaryUp);
      el.removeEventListener('tertiary-touch-down', onTertiaryDown);
      el.removeEventListener('tertiary-touch-up', onTertiaryUp);
      if (timeOutRef.current) clearTimeout(timeOutRef.current);
      console.log('Cleanup called, clearing timeout');
    };
  }, [delay, stopPropagation]);

  return (
    <div
      data-can-interact=''
      {...props}
      ref={divRef}
      className={cn('flex select-none', props.className)}
      style={{
        boxShadow: showFeedback
          ? selected !== -1
            ? `inset 0 0 40px ${colors[selected as keyof typeof colors]}`
            : selecting
              ? 'inset 0 0 40px white'
              : 'none'
          : 'none',
        transition: 'all 0.2s ease-in-out',
        ...props.style
      }}
    >
      {children}
    </div>
  );
}
