// On the web: when the page is served by llmman serve, that daemon is the
// server, with no connect screen. A 401 from llmman's own key check asks
// for the key, as llmman's web UI does.

import { testConnection } from "@/api/connection";
import { importWebUiChats } from "@/lib/conversations";
import { askPageKey, pageKey, pageUrl, setPageKey } from "@/lib/page";
import { getServerKey, saveServer, type ServerList } from "@/lib/servers";

export async function adoptPageServer(list: ServerList): Promise<ServerList> {
  const url = pageUrl();
  if (!url) return list;
  const known = list.servers.find((s) => s.url === url);
  let key = pageKey() || (known && (await getServerKey(known))) || undefined;
  let result = await testConnection(url, key, { timeoutMs: 4000 }).catch(() => null);
  if (result && !result.ok && result.problem === "unauthorized") {
    key = (await askPageKey()) ?? key;
    result = await testConnection(url, key, { timeoutMs: 4000 }).catch(() => null);
  }
  // Not llmman serve (a development server, say): the usual setup applies.
  if (!result || (!result.ok && result.problem !== "unauthorized")) return list;
  // The page's key is the one requests send (page.web.ts): keep it in step.
  if (key && key !== pageKey()) setPageKey(key);
  const name = known?.name ?? new URL(url).hostname;
  const next = await saveServer(list, { url, name, ...(key ? { key } : {}) }, { activate: true });
  // The built-in web UI's chats on this page were with this daemon: they
  // come along, once (its database is left as it is).
  void importWebUiChats(url).catch(() => {});
  return next;
}
