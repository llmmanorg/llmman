// Renders lib/markdown.ts's tree with Text and View only: no HTML, no
// WebView, no images. A link asks before leaving the app. Blocks are
// memoized by their source, so while a reply streams only the last block
// re-renders.

import * as Clipboard from "expo-clipboard";
import { memo, useMemo, useState } from "react";
import {
  Linking,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  type TextStyle,
} from "react-native";

import { isSafeUrl, parseInline, parseMarkdown, type Block, type Inline } from "@/lib/markdown";
import { font, radius, spacing, useTheme, type Palette } from "@/lib/theme";
import { alert } from "@/lib/alert";

export function Markdown({ source, color }: { source: string; color?: string }) {
  const blocks = useMemo(() => parseMarkdown(source), [source]);
  return (
    <View style={styles.root}>
      {blocks.map((b, i) => (
        <BlockView key={i} block={b} raw={b.raw} color={color} />
      ))}
    </View>
  );
}

/** Opens a link from a model's reply after the user confirms the address. */
export function openLink(url: string) {
  if (!isSafeUrl(url)) return;
  alert("Open link?", url, [
    { text: "Cancel", style: "cancel" },
    { text: "Open", onPress: () => void Linking.openURL(url).catch(() => {}) },
  ]);
}

const BlockView = memo(
  function BlockView({ block, color }: { block: Block; raw: string; color?: string }) {
    const c = useTheme();
    return renderBlock(block, c, color ?? c.text);
  },
  (a, b) => a.raw === b.raw && a.color === b.color,
);

function renderBlock(block: Block, c: Palette, color: string): React.ReactElement {
  const base: TextStyle = { color, fontSize: font.body, lineHeight: Math.round(font.body * 1.5) };
  switch (block.type) {
    case "paragraph":
      return (
        <Text style={base} selectable>
          <Inlines nodes={parseInline(block.text)} c={c} />
        </Text>
      );
    case "heading": {
      const size = [0, 24, 21, 19, 17, 16, 16][block.level] ?? 16;
      return (
        <Text
          style={[base, { fontSize: size, lineHeight: size * 1.3, fontWeight: "700" }]}
          accessibilityRole="header"
        >
          <Inlines nodes={parseInline(block.text)} c={c} />
        </Text>
      );
    }
    case "hr":
      return <View style={[styles.hr, { backgroundColor: c.border }]} />;
    case "code":
      return <CodeBlock lang={block.lang} text={block.text} c={c} />;
    case "quote":
      return (
        <View style={[styles.quote, { borderLeftColor: c.border }]}>
          {block.children.map((b, i) => (
            <View key={i}>{renderBlock(b, c, c.textMuted)}</View>
          ))}
        </View>
      );
    case "list":
      return (
        <View style={styles.list}>
          {block.items.map((item, i) => (
            <View key={i} style={styles.item}>
              <Text style={[base, styles.marker]}>{block.ordered ? `${block.start + i}.` : "•"}</Text>
              <View style={styles.itemBody}>
                {item.map((b, j) => (
                  <View key={j}>{renderBlock(b, c, color)}</View>
                ))}
              </View>
            </View>
          ))}
        </View>
      );
    case "table":
      return (
        <ScrollView horizontal style={[styles.table, { borderColor: c.border }]}>
          <View>
            {[block.header, ...block.rows].map((row, r) => (
              <View
                key={r}
                style={[
                  styles.tr,
                  { borderTopColor: c.border, borderTopWidth: r ? StyleSheet.hairlineWidth : 0 },
                ]}
              >
                {block.header.map((_, k) => (
                  <Text
                    key={k}
                    selectable
                    style={[
                      base,
                      styles.td,
                      { textAlign: block.aligns[k] || "left", fontWeight: r === 0 ? "700" : "400" },
                    ]}
                  >
                    <Inlines nodes={parseInline(row[k] ?? "")} c={c} />
                  </Text>
                ))}
              </View>
            ))}
          </View>
        </ScrollView>
      );
  }
}

function Inlines({ nodes, c }: { nodes: Inline[]; c: Palette }) {
  return (
    <>
      {nodes.map((n, i) => {
        switch (n.type) {
          case "text":
            return n.text;
          case "br":
            return "\n";
          case "code":
            return (
              <Text
                key={i}
                style={{ fontFamily: font.mono, backgroundColor: c.codeBackground, fontSize: font.body - 2 }}
              >
                {n.text}
              </Text>
            );
          case "strong":
            return (
              <Text key={i} style={{ fontWeight: "700" }}>
                <Inlines nodes={n.children} c={c} />
              </Text>
            );
          case "em":
            return (
              <Text key={i} style={{ fontStyle: "italic" }}>
                <Inlines nodes={n.children} c={c} />
              </Text>
            );
          case "del":
            return (
              <Text key={i} style={{ textDecorationLine: "line-through" }}>
                <Inlines nodes={n.children} c={c} />
              </Text>
            );
          case "link":
            return (
              <Text
                key={i}
                style={{ color: c.accent, textDecorationLine: "underline" }}
                onPress={() => openLink(n.url)}
                accessibilityRole="link"
                accessibilityHint={n.url}
              >
                <Inlines nodes={n.children} c={c} />
              </Text>
            );
        }
      })}
    </>
  );
}

function CodeBlock({ lang, text, c }: { lang: string; text: string; c: Palette }) {
  const [copied, setCopied] = useState(false);
  // Widths and offset, for the fade that says more code is out of view.
  const [box, setBox] = useState(0);
  const [inner, setInner] = useState(0);
  const [x, setX] = useState(0);
  const copy = async () => {
    await Clipboard.setStringAsync(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <View style={[styles.code, { backgroundColor: c.codeBackground, borderColor: c.border }]}>
      <View style={[styles.codeHead, { borderBottomColor: c.border }]}>
        <Text style={{ color: c.textMuted, fontSize: font.small }}>{lang || "text"}</Text>
        <Pressable onPress={copy} accessibilityRole="button" accessibilityLabel="Copy code" hitSlop={8}>
          <Text style={{ color: c.accent, fontSize: font.small, fontWeight: "600" }}>
            {copied ? "Copied" : "Copy"}
          </Text>
        </Pressable>
      </View>
      <View>
        <ScrollView
          horizontal
          onLayout={(e) => setBox(e.nativeEvent.layout.width)}
          onContentSizeChange={(w) => setInner(w)}
          onScroll={(e) => setX(e.nativeEvent.contentOffset.x)}
          scrollEventThrottle={32}
        >
          <Text
            selectable
            style={{
              color: c.text,
              fontFamily: font.mono,
              fontSize: 14,
              lineHeight: 21,
              padding: spacing.md,
            }}
          >
            {text}
          </Text>
        </ScrollView>
        {/* A fade at an edge with more code past it: the block scrolls sideways. */}
        {x > 1 ? <Fade side="left" color={c.codeBackground} /> : null}
        {inner > box + 1 && x < inner - box - 1 ? <Fade side="right" color={c.codeBackground} /> : null}
      </View>
    </View>
  );
}

/** An edge fading into the code's background (a gradient on the web, steps elsewhere). */
function Fade({ side, color }: { side: "left" | "right"; color: string }) {
  const edge = side === "left" ? { left: 0 } : { right: 0 };
  if (Platform.OS === "web") {
    return (
      <View
        pointerEvents="none"
        style={[
          styles.fade,
          edge,
          {
            backgroundImage: `linear-gradient(to ${side}, transparent, ${color})`,
          } as object,
        ]}
      />
    );
  }
  const steps = [0.15, 0.35, 0.6, 0.85];
  return (
    <View
      pointerEvents="none"
      style={[styles.fade, edge, { flexDirection: side === "left" ? "row-reverse" : "row" }]}
    >
      {steps.map((o) => (
        <View key={o} style={{ flex: 1, backgroundColor: color, opacity: o }} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { gap: spacing.sm },
  hr: { height: StyleSheet.hairlineWidth, marginVertical: spacing.sm },
  quote: { borderLeftWidth: 3, paddingLeft: spacing.md, gap: spacing.sm },
  list: { gap: spacing.xs },
  item: { flexDirection: "row" },
  marker: { minWidth: 22, paddingRight: 4 },
  itemBody: { flex: 1, gap: spacing.xs },
  table: { borderWidth: StyleSheet.hairlineWidth, borderRadius: radius.sm },
  tr: { flexDirection: "row" },
  td: { minWidth: 90, maxWidth: 260, paddingHorizontal: spacing.sm, paddingVertical: spacing.xs },
  code: { borderWidth: StyleSheet.hairlineWidth, borderRadius: radius.md, overflow: "hidden" },
  fade: { position: "absolute", top: 0, bottom: 0, width: 28 },
  codeHead: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: spacing.md,
    paddingVertical: 6,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
});
