// A model's card as a screen of its own (phones; see ModelCard.tsx).

import { Stack, useLocalSearchParams } from "expo-router";

import { shortName } from "@/components/models/logic";
import { ModelCardView } from "@/components/models/ModelCard";

export default function Card() {
  const { name = "" } = useLocalSearchParams<{ name: string }>();
  return (
    <>
      <Stack.Screen options={{ title: shortName(name) }} />
      <ModelCardView name={name} />
    </>
  );
}
