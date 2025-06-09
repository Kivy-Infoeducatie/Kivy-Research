import { RefObject } from 'react';

export function eventPropagation(
  hoveredElements: RefObject<Set<Element>>,
  x: number,
  y: number,
  eventType: 'primary-touch' | 'secondary-touch' | 'tertiary-touch'
) {
  const elementsAtPoint = document
    .elementsFromPoint(x, y)
    .filter((el) => el.getAttribute('data-can-interact') === '');

  const currentElementsSet = new Set(elementsAtPoint);

  elementsAtPoint.forEach((element) => {
    if (!hoveredElements.current.has(element)) {
      element.dispatchEvent(
        new CustomEvent(`${eventType}-down`, {
          bubbles: true,
          cancelable: true
        })
      );
    }
  });

  hoveredElements.current.forEach((element) => {
    if (!currentElementsSet.has(element)) {
      element.dispatchEvent(
        new CustomEvent(`${eventType}-up`, {
          bubbles: true,
          cancelable: true
        })
      );
    }
  });

  hoveredElements.current = currentElementsSet;
}
