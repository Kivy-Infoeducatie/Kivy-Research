export type TimerStack = Timer[];

export interface Timer {
  id: string;
  title: string;
  totalTime: number;
  currentTime: number;
  isRunning: boolean;
}
