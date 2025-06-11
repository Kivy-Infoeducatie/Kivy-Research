import { ReactNode } from 'react';
import Selectable from '@/components/playground/core/selectable';
import { cn } from '@/lib/utils';

interface MainSelectableProps {
  onPress: () => void;
  title?: string;
  icon?: ReactNode;
  showBack?: boolean;
}

export default function MainSelectable({
  onPress,
  title,
  icon,
  showBack = false
}: MainSelectableProps) {
  return (
    <Selectable
      onPrimaryPress={onPress}
      className={cn(
        'relative z-10 flex h-72 w-72 flex-col items-center justify-center rounded-full bg-white text-4xl text-white'
      )}
    >
      {icon && <div className='absolute inset-0 flex items-center'>{icon}</div>}
      <img
        src='/mesh-gradient.png'
        alt='Kivy Logo'
        className='absolute min-h-[45rem] min-w-[45rem]'
      />
      {title ? (
        <span className='text-6xl font-bold text-black'>{title}</span>
      ) : (
        <img src='/kivy-logo.png' alt='Kivy Logo' className='size-52' />
      )}
      {showBack && (
        <div className='mt-5 flex items-center justify-center gap-3'>
          <i className='fa fa-arrow-left text-4xl text-black'></i>
          <span className='text-4xl font-bold text-black'>Back</span>
        </div>
      )}
    </Selectable>
  );
}
