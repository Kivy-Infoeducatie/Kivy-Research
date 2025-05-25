import { ReactNode, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import Selectable from '../core/selectable.tsx';
import { cn } from '../../lib/utils.ts';
import MeasureOverlay from '../overlays/measure-overlay.tsx';
import TimerOverlay from './timer/timer-overlay.tsx';

export default function () {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => setIsOpen(!isOpen);

  const radius = 300;

  const [state, setState] = useState<boolean>(false);
  const [state2, setState2] = useState<boolean>(false);
  const [time, setTime] = useState(0);
  const [countdown, setCountdown] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    let timer;
    if (isRunning && countdown > 0) {
      console.log(countdown);
      timer = setInterval(() => {
        setCountdown((prev) => prev - 1);
      }, 1000);
    } else if (countdown === 0) {
      setIsRunning(false);
    }
    return () => clearInterval(timer);
  }, [isRunning, countdown]);

  const startTimer = (time: number) => {
    console.log(time);
    if (time > 0) {
      setTime(time);
      setCountdown(time);
      setIsRunning(true);
    }
  };

  const stopTimer = () => {
    setIsRunning(false);
  };

  const menuItems: {
    id: number;
    label: string;
    icon: ReactNode;
    onPress?(): void;
  }[] = [
    {
      id: 1,
      label: 'Option 1',
      icon: <i className='fa text-6xl fa-timer' />,
      onPress() {
        setState(false);
        setState2(true);
      }
    },
    {
      id: 2,
      label: 'Option 2',
      icon: <i className='fa text-6xl fa-ruler' />,
      onPress() {
        setState(true);
        setState2(false);
      }
    },
    { id: 3, label: 'Option 3', icon: <i className='fa text-6xl fa-knife' /> },
    {
      id: 4,
      label: 'Option 4',
      icon: <i className='fa text-6xl fa-home' />,
      onPress() {
        setState(false);
        setState2(false);
      }
    },
    { id: 5, label: 'Option 5', icon: <i className='fa text-6xl fa-brain' /> }
  ];

  return (
    <>
      <div className='relative w-80 h-80 flex items-center justify-center z-[100]'>
        <Selectable
          onPress={toggleMenu}
          className={cn(
            'w-80 h-80 bg-red-500 rounded-full flex items-center justify-center text-white z-10 text-4xl'
          )}
        >
          {countdown === 0
            ? 'Nothing selected'
            : `${('00' + Math.floor(countdown / 60)).slice(-2)}:${('00' + (countdown % 60)).slice(-2)}`}
        </Selectable>
        {menuItems.map((item, index) => {
          const angle =
            -Math.PI * (0.4 + index / (menuItems.length - 1) / 1.43);
          const x = radius * Math.cos(angle);
          const y = radius * Math.sin(angle);

          return (
            <motion.div
              key={item.id}
              className='absolute w-32 h-32'
              initial={{ x: 0, y: 0, opacity: 0 }}
              animate={{
                x: isOpen ? x : 0,
                y: isOpen ? y : 0,
                opacity: isOpen ? 1 : 0
              }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            >
              <Selectable
                onPress={() => {
                  if (item.onPress) {
                    item.onPress();
                    toggleMenu();
                  }
                }}
                className='w-32 h-32 bg-blue-500 rounded-full flex items-center justify-center text-white'
              >
                {item.icon}
              </Selectable>
            </motion.div>
          );
        })}
      </div>
      <MeasureOverlay open={state} />
      <TimerOverlay
        open={state2}
        setTime={startTimer}
        setOpen={setState2}
        time={time}
      />
    </>
  );
}
