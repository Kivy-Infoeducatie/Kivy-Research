'use client';

import { useMouseSupport } from '@/lib/core/hand-tracking/use-mouse-support';
import { HandTrackingProvider } from '@/lib/core/hand-tracking/hand-tracking-context';
import { MultiProvider } from '@/components/playground/misc/multi-provider';
import { WidgetsProvider } from '@/lib/core/widgets/widget-context';
import { ScreenProvider } from '@/lib/core/screens/screen-context';

export default function () {
  useMouseSupport();

  return (
    <MultiProvider providers={[HandTrackingProvider, WidgetsProvider]}>
      <ScreenProvider />
    </MultiProvider>
  );
}
