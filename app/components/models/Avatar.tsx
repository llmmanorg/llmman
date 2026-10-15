// A repo owner's picture, or the model family's monogram on a stable
// per-family colour when it has none (or until it loads).

import { useEffect, useState } from "react";
import { Image, StyleSheet, Text, View } from "react-native";

import type { ApiClient } from "@/api/client";
import { avatarUrl } from "@/api/search";
import { useServers } from "@/lib/ServersContext";
import { repoName } from "@/lib/modelName";
import { ownerOf, registryOf, repoOf } from "./logic";

// One lookup per server and repo for the app's lifetime; the daemon itself
// caches the answer for a day.
const lookups = new Map<string, Promise<string | null>>();

function lookup(client: ApiClient, repo: string): Promise<string | null> {
  const key = `${client.url}\n${repo}`;
  let p = lookups.get(key);
  if (!p) {
    p = avatarUrl(client, repo).catch(() => {
      lookups.delete(key); // a network error is worth another try later
      return null;
    });
    lookups.set(key, p);
  }
  return p;
}

export function Avatar({ name, size = 36 }: { name: string; size?: number }) {
  const { client } = useServers();
  const repo = repoOf(name);
  // Both keyed by what they are for, so a recycled row never shows the last repo's picture.
  const key = `${client?.url}\n${repo}`;
  const [found, setFound] = useState<{ key: string; uri: string | null } | null>(null);
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const uri = found?.key === key ? found.uri : null;
  const loaded = !!uri && loadedKey === key;

  // Docker's own `ai` models all carry the Docker logo: a list of identical
  // pictures says nothing, so they get the family's monogram instead.
  const shared = registryOf(repo) === "docker" && ownerOf(repo) === "ai";
  useEffect(() => {
    let live = true;
    if (client && !shared) lookup(client, repo).then((u) => live && setFound({ key, uri: u }));
    return () => {
      live = false;
    };
  }, [client, repo, key, shared]);

  // Without a picture: the model family's monogram on its own colour (a
  // publisher's would be the same for every model it publishes).
  const family = repoName(repo).split(/[\s\d.]/)[0] || "?";
  const mono = family.slice(0, 2);
  const hue = [...family.toLowerCase()].reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) % 360, 7);
  const box = { width: size, height: size, borderRadius: size / 4 };
  return (
    <View
      style={[styles.box, box, { backgroundColor: `hsl(${hue}, 45%, 42%)` }]}
      accessible={false}
      importantForAccessibility="no-hide-descendants"
    >
      {loaded ? null : (
        <Text style={[styles.initials, { fontSize: size * 0.38 }]} allowFontScaling={false}>
          {mono}
        </Text>
      )}
      {uri ? (
        <Image
          source={{ uri }}
          style={[StyleSheet.absoluteFill, box, { opacity: loaded ? 1 : 0 }]}
          onLoad={() => setLoadedKey(key)}
          onError={() => setFound({ key, uri: null })}
        />
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  box: { alignItems: "center", justifyContent: "center", overflow: "hidden" },
  initials: { color: "#FFFFFF", fontWeight: "700" },
});
