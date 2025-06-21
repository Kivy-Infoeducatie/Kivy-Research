import { MenuItems } from '@/components/playground/widgets/home-widget/menu-items';
import MainSelectable from '@/components/playground/widgets/home-widget/main-selectable';
import { useState } from 'react';

const createHomeMenuItems = (
  onSelectWidget: (widgetId: string, title: string) => void
) => [
  {
    id: 1,
    label: 'Timer',
    icon: <i className='fa fa-timer text-6xl' />,
    onPress() {
      onSelectWidget('timer', 'Timer');
    }
  },
  {
    id: 2,
    label: 'Measure',
    icon: <i className='fa fa-ruler text-6xl' />,
    onPress() {
      onSelectWidget('measure', 'Measure');
    }
  },
  {
    id: 3,
    label: 'Cutting',
    icon: <i className='fa fa-knife text-6xl' />,
    onPress() {
      onSelectWidget('cutting', 'Cutting');
    }
  },
  {
    id: 4,
    label: 'Recipes',
    icon: <i className='fa fa-book text-6xl' />,
    onPress() {
      onSelectWidget('recipes', 'Recipes');
    }
  },
  {
    id: 5,
    label: 'AI',
    icon: <i className='fa fa-brain text-6xl' />,
    onPress() {
      onSelectWidget('ai', 'AI Assistant');
    }
  }
];

export function HomeWidget() {
  const [isOpen, setIsOpen] = useState(false);

  const handleMainPress = () => {
    setIsOpen(!isOpen);
  };

  const getMenuItems = () => {
    return createHomeMenuItems(() => {});
  };

  return (
    <div className='fixed right-12 bottom-12 z-[100] flex h-80 w-80 items-center justify-center'>
      <MainSelectable
        title={'Home'}
        onPress={handleMainPress}
        icon={
          <i className='fa fa-arrow-left absolute top-1/2 left-8 -translate-y-1/2 transform text-xl text-white' />
        }
        showBack={false}
      />

      <MenuItems
        isOpen={isOpen}
        menuItems={getMenuItems()}
        setIsOpen={setIsOpen}
      />
      <img
        src='/mesh-gradient.png'
        alt='Kivy Logo'
        className='pointer-events-none absolute z-[100] min-h-[45rem] min-w-[45rem]'
      />
    </div>
  );
}
