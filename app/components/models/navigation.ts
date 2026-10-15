import { router } from "expo-router";

// On a wide screen the Models screen shows the card beside the list; while
// it does, it takes the card to show here instead of a pushed screen.
let showInPane: ((name: string) => void) | null = null;

/** Registers the wide Models screen's card pane; returns the unregister. */
export function setCardPane(show: (name: string) => void): () => void {
  showInPane = show;
  return () => {
    if (showInPane === show) showInPane = null;
  };
}

/** Opens repository `name`'s card on the Models screen. */
export function openModel(name: string) {
  if (showInPane) showInPane(name);
  else router.push({ pathname: "/models/[name]", params: { name } });
}

/** Opens the version picker for repository `name`'s card. */
export function openVariants(name: string) {
  router.push({ pathname: "/models/variants", params: { name } });
}

/** The Models page's width on a wide screen: list and card, centred. */
export const MODELS_MAX_WIDTH = 1180;
