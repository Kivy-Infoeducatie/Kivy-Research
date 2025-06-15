import { useEffect } from 'react';
import { dispatchEvent } from '@/lib/core/hand-tracking/event-propagation';

function dispatchEventAtPoint(x: number, y: number, eventType: string) {
  const elementsAtPoint = document
    .elementsFromPoint(x, y)
    .filter((el) => el.getAttribute('data-can-interact') === '');

  const currentElementsSet = new Set(elementsAtPoint);

  currentElementsSet.forEach((element) => {
    dispatchEvent(element, x, y, eventType, -1);
  });
}

export function useMouseSupport() {
  useEffect(() => {
    function onMouseDown(event: MouseEvent) {
      if (event.metaKey || event.button === 1) {
        dispatchEventAtPoint(
          event.clientX,
          event.clientY,
          'primary-touch-down'
        );
      } else if (event.button === 2) {
        dispatchEventAtPoint(
          event.clientX,
          event.clientY,
          'secondary-touch-down'
        );
      } else {
        dispatchEventAtPoint(
          event.clientX,
          event.clientY,
          'tertiary-touch-down'
        );
      }
    }

    function onMouseUp(event: MouseEvent) {
      if (event.metaKey || event.button === 1) {
        dispatchEventAtPoint(event.clientX, event.clientY, 'primary-touch-up');
      } else if (event.button === 2) {
        dispatchEventAtPoint(
          event.clientX,
          event.clientY,
          'secondary-touch-up'
        );
      } else {
        dispatchEventAtPoint(event.clientX, event.clientY, 'tertiary-touch-up');
      }
    }

    function preventDefaultContextMenu(e: MouseEvent) {
      e.preventDefault();
    }

    document.addEventListener('contextmenu', preventDefaultContextMenu);

    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mouseup', onMouseUp);

    return () => {
      document.removeEventListener('contextmenu', preventDefaultContextMenu);

      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }, []);
}
