// A saved conversation.

import { useLocalSearchParams } from "expo-router";

import { ChatScreen } from "@/components/chat/ChatScreen";

export default function Conversation() {
  const { id } = useLocalSearchParams<{ id: string }>();
  return <ChatScreen key={id} id={id} />;
}
