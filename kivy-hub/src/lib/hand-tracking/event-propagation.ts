import { MutableRefObject } from 'react';

export function eventPropagation(
  hoveredElements: MutableRefObject<Set<Element>>,
  x: number,
  y: number
) {
  const elementsAtPoint = document
    .elementsFromPoint(x, y)
    .filter((el) => el.getAttribute('data-can-interact') === '');

  const currentElementsSet = new Set(elementsAtPoint);

  elementsAtPoint.forEach((element) => {
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

  hoveredElements.current = currentElementsSet;
}
