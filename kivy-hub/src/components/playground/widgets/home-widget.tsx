import { useEffect, useState } from 'react';
import MainSelectable from './main-selectable';
import MenuItems from './menu-items';
import useTimerMenu from './timer/timer-menu';

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

export default function HomeWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const { screenStack, activeWidget, pushScreen, popScreen } = useWidget();
  const timerMenuItems = useTimerMenu();

  const currentScreen = screenStack[screenStack.length - 1];
  const isRootScreen = screenStack.length === 1;

  useEffect(() => {
    if (!isRootScreen) {
      setIsOpen(true);
    }
  }, [isRootScreen, screenStack]);

  const handleMainPress = () => {
    if (isRootScreen) {
      setIsOpen(!isOpen);
    } else {
      popScreen();
    }
  };

  const handleSelectWidget = (widgetId, title) => {
    pushScreen({ id: widgetId, title });
    setIsOpen(true);
  };

  const getMenuItems = () => {
    switch (activeWidget) {
      case 'timer':
        return timerMenuItems;
      default:
        return createHomeMenuItems(handleSelectWidget);
    }
  };

  const displayIcon = !isRootScreen ? (
    <i className='fa fa-arrow-left absolute top-1/2 left-8 -translate-y-1/2 transform text-xl text-white'></i>
  ) : null;

  const shouldShowMenu = !isRootScreen || isOpen;

  return (
    <div className='relative z-[100] flex h-80 w-80 items-center justify-center'>
      <MainSelectable
        title={currentScreen.title}
        onPress={handleMainPress}
        icon={displayIcon}
        showBack={!isRootScreen}
      />

      <MenuItems
        isOpen={shouldShowMenu}
        menuItems={getMenuItems()}
        toggleMenu={() => {
          if (isRootScreen) {
            setIsOpen(!isOpen);
          }
        }}
      />
    </div>
  );
}
