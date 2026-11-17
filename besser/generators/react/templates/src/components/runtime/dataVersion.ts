import { useSyncExternalStore } from "react";

// Counts the records saved or removed through the runtime components. The
// components that show data reload it when the count changes, so a write made
// in one of them (a table, a form, a method button) reaches all the others.
let version = 0;
const listeners = new Set<() => void>();

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};

/** Call after a successful write to the backend. */
export const notifyDataChanged = (): void => {
  version += 1;
  listeners.forEach((listener) => listener());
};

/** A number that changes whenever the backend data has been written to. */
export const useDataVersion = (): number => useSyncExternalStore(subscribe, () => version);
