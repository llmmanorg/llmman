// One header for the app's pages on a desktop browser (Models, Providers,
// Settings, Servers, Shell): the title aligned with the page's
// centred column, the page's actions inset from the window's edge, and the
// menu (a collapsed sidebar) or back button at the left. Phones and narrow
// windows keep the platform's own header.

import type { ReactNode } from "react";
import { Platform, Pressable, StyleSheet, Text, useWindowDimensions, View } from "react-native";

import { Icon } from "@/components/Icon";
import { MenuButton, useSidebar } from "@/components/shell/Header";
import { font, readable, spacing, useTheme } from "@/lib/theme";

/** The sidebar's width when docked (app/(drawer)/_layout.tsx). */
const SIDEBAR = 280;
/** Room the menu or back button takes at the left. */
const LEFT = 56;

export interface PageHeaderProps {
  options: {
    title?: string;
    headerRight?: (props: { tintColor?: string; canGoBack: boolean }) => ReactNode;
  };
  back?: unknown;
  navigation: { goBack(): void; canGoBack?(): boolean };
}

/** Where a page's column starts in its content area, so the title lines up with it. */
export function columnInset(contentWidth: number, maxWidth: number): number {
  return Math.max(0, (contentWidth - maxWidth) / 2) + spacing.lg;
}

export function PageHeader({ options, back, navigation, maxWidth }: PageHeaderProps & { maxWidth: number }) {
  const c = useTheme();
  const { width } = useWindowDimensions();
  const { docked } = useSidebar();
  const content = width - (docked ? SIDEBAR : 0);
  const showMenu = !docked;
  const left = Math.max(columnInset(content, maxWidth), back || showMenu ? LEFT : spacing.lg);
  const right = options.headerRight?.({ tintColor: c.text, canGoBack: !!back });
  return (
    <View style={[styles.bar, { backgroundColor: c.background }]}>
      <View style={styles.left}>
        {back ? (
          <Pressable
            onPress={() => navigation.goBack()}
            accessibilityRole="button"
            accessibilityLabel="Back"
            hitSlop={8}
            style={styles.icon}
          >
            {/* The app's chevron, turned: no other icon is needed for it. */}
            <View style={styles.flip}>
              <Icon name={{ ios: "chevron.right", android: "chevron_right" }} color={c.text} size={20} />
            </View>
          </Pressable>
        ) : (
          <MenuButton />
        )}
      </View>
      <Text
        style={[styles.title, { color: c.text, marginLeft: left }]}
        numberOfLines={1}
        accessibilityRole="header"
      >
        {options.title}
      </Text>
      {right ? <View style={styles.right}>{right}</View> : null}
    </View>
  );
}

/**
 * Stack options that give a page the desktop header on a wide web window,
 * and nothing elsewhere. `maxWidth` is the page's column (720 for most).
 */
export function usePageHeader(maxWidth: number = readable.maxWidth): {
  header?: (props: PageHeaderProps) => ReactNode;
} {
  const { wide } = useSidebar();
  if (Platform.OS !== "web" || !wide) return {};
  return { header: (props) => <PageHeader {...props} maxWidth={maxWidth} /> };
}

const styles = StyleSheet.create({
  bar: { height: 56, flexDirection: "row", alignItems: "center" },
  left: { position: "absolute", left: spacing.md, top: 0, bottom: 0, justifyContent: "center", zIndex: 1 },
  icon: { padding: spacing.xs + 2 },
  flip: { transform: [{ rotate: "180deg" }] },
  title: { flex: 1, fontSize: font.title - 2, fontWeight: "600" },
  right: { paddingRight: spacing.xl - 4, flexDirection: "row", alignItems: "center", gap: spacing.sm },
});
