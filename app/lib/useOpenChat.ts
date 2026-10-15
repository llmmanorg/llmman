// Start a new conversation, optionally with a given model chosen (the
// Models screen's "Chat" action): home is always a new chat.

import { router } from "expo-router";
import { useCallback } from "react";

import { useModels } from "@/lib/ModelsContext";

export function useOpenChat(): (model?: string) => void {
  const { select } = useModels();
  return useCallback(
    (model?: string) => {
      if (model) select(model);
      router.navigate("/");
    },
    [select],
  );
}
