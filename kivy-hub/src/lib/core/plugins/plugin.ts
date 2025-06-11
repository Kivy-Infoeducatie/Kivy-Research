export abstract class Plugin {
  abstract onMount(): void;

  abstract onUnmount(): void;
}