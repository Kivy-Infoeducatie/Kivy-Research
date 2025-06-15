import { RefObject } from 'react';
import { HandEvent } from '@/lib/core/hand-tracking/hand-tracking-types';

const handEvents = {
  [-1]: 'no-touch',
  [0]: 'primary-touch',
  [1]: 'secondary-touch',
  [2]: 'tertiary-touch'
};

export function dispatchEvent(
  element: Element,
  x: number,
  y: number,
  eventType: string,
  handIndex: number,
  type = 'hand'
) {
  element.dispatchEvent(
    new CustomEvent(eventType, {
      bubbles: true,
      cancelable: true,
      detail: {
        clientX: x,
        clientY: y,
        type,
        handIndex
      }
    })
  );
}

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
      dispatchEvent(element, x, y, handEvents[eventType] + '-down', handIndex);
    }
  });

  hoveredElements.current.forEach((element) => {
    if (!currentElementsSet.has(element)) {
      dispatchEvent(element, x, y, handEvents[eventType] + '-up', handIndex);
    } else if (hoveredElementEvents.current.get(element) !== eventType) {
      dispatchEvent(
        element,
        x,
        y,
        handEvents[hoveredElementEvents.current.get(element)!] + '-up',
        handIndex
      );
      hoveredElementEvents.current.delete(element);
    }
  });

  hoveredElements.current = currentElementsSet;
}
