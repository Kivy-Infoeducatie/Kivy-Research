import Overlay from '../../core/overlay.tsx';
import Selectable from '../../core/selectable.tsx';

export default function ({
  open,
  setTime,
  setOpen,
  time
}: {
  open: boolean;
  setTime: (value: number) => void;
  setOpen: (value: boolean) => void;
  time: number;
}) {
  return (
    <Overlay
      open={open}
      className='bg-black/90 flex items-center justify-center gap-12'
    >
      <Selectable
        onPress={() => {
          setTime(5 * 60);
          setOpen(false);
        }}
        className='rounded-full bg-amber-500 w-40 h-40 flex justify-center items-center text-4xl'
      >
        5 min
      </Selectable>
      <Selectable
        onPress={() => {
          setTime(15 * 60);
          setOpen(false);
        }}
        className='rounded-full bg-amber-500 w-40 h-40 flex justify-center items-center text-4xl'
      >
        15 min
      </Selectable>
      <Selectable
        onPress={() => {
          setTime(30 * 60);
          setOpen(false);
        }}
        className='rounded-full bg-amber-500 w-40 h-40 flex justify-center items-center text-4xl'
      >
        30 min
      </Selectable>
    </Overlay>
  );
}
