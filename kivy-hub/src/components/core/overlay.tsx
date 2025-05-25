import { HTMLAttributes } from 'react';
import { cn } from '../../lib/utils.ts';

export default function ({
  open = false,
  ...props
}: HTMLAttributes<HTMLDivElement> & {
  open?: boolean;
}) {
  return (
    <div
      {...props}
      style={{
        display: open ? 'flex' : 'none',
        ...props.style
      }}
      className={cn(
        'fixed top-0 left-0 w-screen h-screen z-50',
        props.className
      )}
    />
  );
}
