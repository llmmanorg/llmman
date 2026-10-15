// Home: a new chat. Once its first message is sent it moves to its own
// route (/c/<id>) and home starts another, so home is always a new chat.

import { router } from "expo-router";
import { useEffect, useState } from "react";
import { View } from "react-native";

import { ChatScreen } from "@/components/chat/ChatScreen";
import { getConversationState, openConversation, setModel } from "@/lib/chatStore";
import { newConversation } from "@/lib/conversations";
import { useModels } from "@/lib/ModelsContext";
import { useServers } from "@/lib/ServersContext";
import { useTheme } from "@/lib/theme";

export default function Home() {
  const c = useTheme();
  const { active } = useServers();
  const { selected } = useModels();
  const [id, setId] = useState<string | null>(null);
  const [round, setRound] = useState(0);

  // A fresh conversation for the active server (it is only saved once it has a message).
  useEffect(() => {
    if (!active) return;
    const conv = newConversation(active.url, selected, active.chatDefaults);
    let cancelled = false;
    void openConversation(conv.id, conv).then(() => !cancelled && setId(conv.id));
    return () => {
      cancelled = true;
    };
    // Only a new server or a sent chat starts another; the model follows below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active?.url, round]);

  // Until it has a message, the new chat uses whichever model was chosen last.
  useEffect(() => {
    if (id && selected && !getConversationState(id).conv?.messages.length) setModel(id, selected);
  }, [id, selected]);

  if (!id) return <View style={{ flex: 1, backgroundColor: c.background }} />;
  return (
    <ChatScreen
      key={id}
      id={id}
      onSent={() => {
        router.navigate({ pathname: "/c/[id]", params: { id } });
        setRound((r) => r + 1);
      }}
    />
  );
}
