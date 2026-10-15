// A model's card: its versions with sizes and how each fits this computer,
// Pull, what this computer already holds of it, and its README in full.

import { useEffect } from "react";
import { ActivityIndicator, Linking, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";

import type { ModelCard, Variant } from "@/api/search";
import { Icon } from "@/components/Icon";
import { Avatar } from "@/components/models/Avatar";
import { Chip, Muted, RegistryBadge, SectionTitle, useTextSizeKey } from "@/components/models/bits";
import { Stat } from "@/components/models/HitRow";
import { LocalTagRow } from "@/components/models/LocalTagRow";
import {
  cardTags,
  counted,
  fitSentence,
  formatBytes,
  formatCount,
  holds,
  chosenVariant,
  localOf,
  registryName,
  registryOf,
  relativeTime,
  shortName,
  variantLabel,
} from "@/components/models/logic";
import { pullOf, useModelsTab } from "@/components/models/ModelsTabContext";
import { openVariants } from "@/components/models/navigation";
import { ErrorBox } from "@/components/models/ErrorBox";
import { PullPanel } from "@/components/models/PullViews";
import { VariantLine } from "@/components/models/VariantLine";
import { Readme } from "@/components/models/Readme";
import { Button } from "@/components/ui";
import { font, radius, spacing, useTheme } from "@/lib/theme";

/** Repository `name`'s card: its own screen on phones, the right pane on wide screens. */
export function ModelCardView({ name }: { name: string }) {
  const c = useTheme();
  const models = useModelsTab();
  const { local, memory, pull, loadCard } = models;
  const textSizeKey = useTextSizeKey();
  const searchable = registryOf(name) !== "other";
  const entry = searchable ? models.card(name) : undefined;
  const info = entry && typeof entry === "object" && !(entry instanceof Error) ? entry : null;

  useEffect(() => {
    if (searchable) loadCard(name);
  }, [searchable, name, loadCard]);

  const held = localOf(name, local);
  const finished = models.noticesFor(name);
  const pullingHere = pullOf(pull, name);
  const chosen = info ? chosenVariant(info.variants, memory, models.picked(name)) : undefined;

  return (
    // The scroll view is the screen's root, so the native tab bar tracks it.
    <ScrollView
      key={textSizeKey}
      style={{ backgroundColor: c.background }}
      contentContainerStyle={styles.content}
      contentInsetAdjustmentBehavior="automatic"
    >
      <Header name={name} info={info} />

      {held.length || finished.length ? (
        <>
          <SectionTitle>On this computer</SectionTitle>
          {held.map((m) => (
            <LocalTagRow key={m.id} model={m} />
          ))}
          {finished.flatMap(({ ref, notices }) =>
            notices.map((n, i) => (
              <Text key={`${ref}-${i}`} style={[styles.notice, { color: c.textMuted }]}>
                {n}
              </Text>
            )),
          )}
        </>
      ) : null}

      {pull && pullingHere ? (
        <PullPanel
          pull={pull}
          onStop={models.stopWatching}
          onRetry={() => {
            models.dismissPull();
            models.startPull(pull.ref);
          }}
          onDismiss={models.dismissPull}
        />
      ) : null}

      {!searchable ? null : entry === "loading" || entry === undefined ? (
        <ActivityIndicator style={styles.loading} accessibilityLabel="Loading the model" />
      ) : entry instanceof Error ? (
        <View style={styles.block}>
          <ErrorBox error={entry} title="Could not load this model" onRetry={() => loadCard(name)} />
        </View>
      ) : (
        <Body name={name} info={entry} chosen={chosen} />
      )}
    </ScrollView>
  );
}

function Header({ name, info }: { name: string; info: ModelCard | null }) {
  const c = useTheme();
  const title = name.split("/").slice(-2).join("/");
  const page = info?.page;
  return (
    <View style={styles.head}>
      <Avatar name={name} size={48} />
      <View style={styles.headText}>
        <Pressable
          onPress={page ? () => Linking.openURL(page) : undefined}
          disabled={!page}
          accessibilityRole={page ? "link" : "text"}
          accessibilityHint={page ? "Opens the model's page on its registry" : undefined}
        >
          <Text style={[styles.title, { color: page ? c.accent : c.text }]}>{title}</Text>
        </Pressable>
        {/* One element for a screen reader: the badge and icons alone say little. */}
        <View
          style={styles.chips}
          accessible
          accessibilityLabel={[
            registryName(name) && `on ${registryName(name)}`,
            info?.pulls != null && counted(info.pulls, "pull"),
            info?.likes != null && counted(info.likes, "like"),
            info?.updated && `updated ${relativeTime(info.updated)}`,
            info?.license && `license ${info.license}`,
            info?.task,
            info?.gated && "gated: needs a Hugging Face token",
          ]
            .filter(Boolean)
            .join(", ")}
        >
          <RegistryBadge name={name} />
          {info?.pulls != null ? <Stat icon="pulls" text={counted(info.pulls, "pull")} /> : null}
          {info?.likes != null ? <Stat icon="likes" text={formatCount(info.likes)} /> : null}
          {info?.updated ? <Chip>{`updated ${relativeTime(info.updated)}`}</Chip> : null}
          {info?.license ? <Chip>{info.license}</Chip> : null}
          {info?.task ? <Chip>{info.task}</Chip> : null}
          {info?.gated ? <Chip tone="accent">gated: needs an HF token</Chip> : null}
        </View>
      </View>
    </View>
  );
}

function Body({ name, info, chosen }: { name: string; info: ModelCard; chosen: Variant | undefined }) {
  const c = useTheme();
  const { local, memory, pull } = useModelsTab();
  const many = info.variants.length > 1;
  const tags = cardTags(info.tags);
  return (
    <>
      {!chosen ? (
        <View style={styles.block}>
          <Muted>Nothing in this repository that llmman can pull.</Muted>
        </View>
      ) : (
        <>
          <SectionTitle>{many ? `Version · ${info.variants.length} to pick from` : "Download"}</SectionTitle>
          <Pressable
            onPress={many ? () => openVariants(name) : undefined}
            disabled={!many}
            accessibilityRole={many ? "button" : "text"}
            accessibilityLabel={`Version ${variantLabel(chosen, local, memory)}`}
            accessibilityHint={many ? "Picks another version" : undefined}
            style={({ pressed }) => [styles.variant, { backgroundColor: pressed ? c.surfaceAlt : c.surface }]}
          >
            <VariantLine
              variant={chosen}
              local={local}
              memory={memory}
              strong
              trail={
                many ? (
                  <Icon
                    name={{ ios: "chevron.up.chevron.down", android: "unfold_more" }}
                    size={16}
                    color={c.textMuted}
                  />
                ) : null
              }
            />
          </Pressable>
          {fitSentence(chosen, memory) ? (
            <Text style={[styles.fit, { color: c.textMuted }]}>{fitSentence(chosen, memory)}</Text>
          ) : null}
          {pullOf(pull, name) ? null : (
            <PullAction
              chosen={chosen}
              have={holds(chosen, local)}
              busy={!!pull && !pull.error && !pull.done}
            />
          )}
        </>
      )}

      {tags.length ? (
        <>
          <SectionTitle>Tags</SectionTitle>
          <View style={styles.chips}>
            {tags.map((t) => (
              <Chip key={t}>{t}</Chip>
            ))}
          </View>
        </>
      ) : null}

      {info.readme ? (
        <>
          <SectionTitle>README</SectionTitle>
          <Readme text={info.readme} page={registryOf(name) === "hf" ? info.page : ""} />
        </>
      ) : null}
    </>
  );
}

/** The card's one action, right under the version it acts on. */
function PullAction({ chosen, have, busy }: { chosen: Variant; have: boolean; busy: boolean }) {
  const c = useTheme();
  const { startPull, pull } = useModelsTab();
  const title = have
    ? `${chosen.tag} is on this computer`
    : `Pull ${chosen.tag}${chosen.size ? ` · ${formatBytes(chosen.size)}` : ""}`;
  return (
    <View style={styles.action}>
      <Button
        title={title}
        disabled={have || busy}
        onPress={() => startPull(chosen.name)}
        icon={
          <Icon
            name={
              have
                ? { ios: "checkmark", android: "check" }
                : { ios: "arrow.down.circle", android: "download" }
            }
            size={18}
            color={have ? c.text : c.onAccent}
          />
        }
        accessibilityHint={busy && pull ? `Another pull is running: ${pull.ref}` : chosen.name}
      />
      {busy && pull ? <Muted>{`Waiting for the pull of ${shortName(pull.ref)} to finish.`}</Muted> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  content: { padding: spacing.lg, paddingBottom: spacing.xl * 2 },
  head: { flexDirection: "row", gap: spacing.md, alignItems: "flex-start" },
  headText: { flex: 1, gap: spacing.sm },
  title: { fontSize: font.title, fontWeight: "700" },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm, alignItems: "center" },
  notice: { fontSize: font.small, lineHeight: 18, marginTop: spacing.xs },
  loading: { marginTop: spacing.xl },
  block: { gap: spacing.md, marginTop: spacing.lg },
  variant: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    minHeight: 52,
    paddingHorizontal: spacing.md,
    borderRadius: radius.md,
  },
  fit: { fontSize: font.small, lineHeight: 18, marginTop: spacing.sm },
  action: { gap: spacing.xs, marginTop: spacing.md },
});
