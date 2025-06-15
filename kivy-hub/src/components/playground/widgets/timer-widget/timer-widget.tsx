import { Timer } from '@/components/playground/widgets/timer-widget/timer-widget-types';
import { Bell, PauseIcon, PlayIcon, TimerIcon } from 'lucide-react';
import { Selectable } from '@/components/playground/core/selectable';

const pad = (num: number): string => num.toString().padStart(2, '0');

function formatSeconds(seconds: number): string {
  const hours: number = Math.floor(seconds / 3600);
  const minutes: number = Math.floor((seconds % 3600) / 60);
  const secs: number = seconds % 60;

  return `${pad(hours)}:${pad(minutes)}:${pad(secs)}`;
}

function getRunOffTime(secondsFromNow: number): string {
  const now = new Date();
  const future = new Date(now.getTime() + secondsFromNow * 1000);

  const hours = future.getHours();
  const minutes = future.getMinutes();

  return `${pad(hours)}:${pad(minutes)}`;
}

export function TimerWidget({
  isRunning,
  totalTime,
  currentTime,
  title,
  id
}: Timer) {
  return (
    <div className='relative flex h-[16rem] w-[23rem] flex-col justify-between rounded-[3rem] bg-white p-5 text-black'>
      <div className='z-10 flex flex-col'>
        <div className='flex items-center gap-2'>
          <TimerIcon className='text-black/80' />
          <h3 className='text-2xl font-bold'>{title}</h3>
        </div>
        <div className='flex flex-col'>
          <label className='text-5xl font-bold'>
            {formatSeconds(currentTime)}
          </label>
          {isRunning && (
            <div className='flex items-center gap-1'>
              <Bell />
              <span>{getRunOffTime(currentTime)}</span>
            </div>
          )}
        </div>
      </div>
      <div className='z-10 flex justify-between'>
        <Selectable
          onPrimaryPress={() => {}}
          className='rounded-3xl bg-amber-400 px-10 py-4'
        >
          <PlayIcon className='size-14' />
        </Selectable>
        <Selectable
          onPrimaryPress={() => {}}
          className='rounded-3xl bg-amber-400 px-10 py-4'
        >
          <PauseIcon className='size-14' />
        </Selectable>
      </div>
      <div
        className='absolute top-0 left-0 h-full rounded-[3rem] bg-green-400'
        style={{
          width: `${(currentTime / totalTime) * 100}%`
        }}
      />
    </div>
  );
}
