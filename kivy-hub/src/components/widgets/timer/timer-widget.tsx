import SelectOverlay from '../../overlays/select-overlay.tsx';
import { useState } from 'react';
import Selectable from '../../core/selectable.tsx';

export default function () {
  const [open, setOpen] = useState<boolean>(false);

  function onSelect() {
    setOpen(false);
  }

  return (
    <>
      <Selectable
        onPress={() => {
          setOpen(true);
        }}
      >
        <div className='w-80 h-80 rounded-full bg-amber-500 flex justify-center items-center'>
          timer
        </div>
      </Selectable>
      <SelectOverlay
        open={open}
        onSelect={onSelect}
        items={[
          {
            label: 'Move',
            value: 'move'
          },
          {
            label: 'Set time',
            value: 'set_time'
          }
        ]}
      />
    </>
  );
}
