import { useEffect } from 'react';
import { HandEvent } from '@/lib/hand-tracking/hand-tracking-context';

const handEvents = {
  [0]: 'primary-touch',
  [1]: 'secondary-touch',
  [2]: 'tertiary-touch'
};

function dispatchEventAtPoint(
  x: number,
  y: number,
  eventType: number,
  eventDirective?: 'down' | 'up'
) {
  const elementsAtPoint = document
    .elementsFromPoint(x, y)
    .filter((el) => el.getAttribute('data-can-interact') === '');

  const currentElementsSet = new Set(elementsAtPoint);

  currentElementsSet.forEach((element) => {
    element.dispatchEvent(
      new CustomEvent(
        `${handEvents[eventType as keyof typeof handEvents]}-${eventDirective}`,
        {
          bubbles: true,
          cancelable: true
        }
      )
    );
  });
}

export function useMouseSupport() {
  useEffect(() => {
    function onMouseDown(event: MouseEvent) {
      if (event.metaKey || event.button === 1) {
        dispatchEventAtPoint(
          event.clientX,
          event.clientY,
          HandEvent.TERTIARY_TOUCH,
          'down'
        );
      } else if (event.button === 2) {
        dispatchEventAtPoint(
          event.clientX,
          event.clientY,
          HandEvent.SECONDARY_TOUCH,
          'down'
        );
      } else {
        dispatchEventAtPoint(
          event.clientX,
          event.clientY,
          HandEvent.PRIMARY_TOUCH,
          'down'
        );
      }
    }

    function onMouseUp(event: MouseEvent) {
      if (event.metaKey || event.button === 1) {
        dispatchEventAtPoint(
          event.clientX,
          event.clientY,
          HandEvent.TERTIARY_TOUCH,
          'up'
        );
      } else if (event.button === 2) {
        dispatchEventAtPoint(
          event.clientX,
          event.clientY,
          HandEvent.SECONDARY_TOUCH,
          'up'
        );
      } else {
        dispatchEventAtPoint(
          event.clientX,
          event.clientY,
          HandEvent.PRIMARY_TOUCH,
          'up'
        );
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
