import { RefObject } from 'react';
import { HandEvent } from '@/lib/core/hand-tracking/hand-tracking-context';

const handEvents = {
  [0]: 'primary-touch',
  [1]: 'secondary-touch',
  [2]: 'tertiary-touch'
};

export function eventPropagation(
  hoveredElements: RefObject<Set<Element>>,
  hoveredElementEvents: RefObject<Map<Element, HandEvent>>,
  x: number,
  y: number,
  eventType: HandEvent,
  handIndex: number
) {
  const elementsAtPoint = document
    .elementsFromPoint(x, y)
    .filter((el) => el.getAttribute('data-can-interact') === '');

  const currentElementsSet = new Set(elementsAtPoint);

  elementsAtPoint.forEach((element) => {
    if (!hoveredElements.current.has(element)) {
      hoveredElementEvents.current.set(element, eventType);
      element.dispatchEvent(
        new CustomEvent(`${handEvents[eventType]}-down`, {
          bubbles: true,
          cancelable: true,
          detail: {
            clientX: x,
            clientY: y,
            type: 'hand',
            handIndex
          }
        })
      );
    }
  });

  hoveredElements.current.forEach((element) => {
    if (!currentElementsSet.has(element)) {
      element.dispatchEvent(
        new CustomEvent(`${handEvents[eventType]}-up`, {
          bubbles: true,
          cancelable: true,
          detail: {
            clientX: x,
            clientY: y,
            type: 'hand',
            handIndex
          }
        })
      );
    } else if (hoveredElementEvents.current.get(element) !== eventType) {
      element.dispatchEvent(
        new CustomEvent(
          `${handEvents[hoveredElementEvents.current.get(element)!]}-up`,
          {
            bubbles: true,
            cancelable: true
          }
        )
      );
      hoveredElementEvents.current.delete(element);
    }
  });

  hoveredElements.current = currentElementsSet;
}
