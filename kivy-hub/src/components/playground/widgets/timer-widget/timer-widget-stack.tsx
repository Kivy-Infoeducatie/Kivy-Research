import { Timer } from '@/components/playground/widgets/timer-widget/timer-widget-types';
import { TimerWidget } from '@/components/playground/widgets/timer-widget/timer-widget';
import { Movable } from '@/components/playground/core/movable';

export function TimerWidgetStack({ timers }: { timers: Timer[] }) {
  return (
    <Movable className='flex flex-col gap-4 rounded-[3rem]'>
      {timers.map((timer) => (
        <TimerWidget key={timer.id} {...timer} />
      ))}
    </Movable>
  );
}
