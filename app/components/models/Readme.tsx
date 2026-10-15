// A repo's README on its card, in full. The registry's HTML is reduced to
// Markdown first (readmeMarkdown), then rendered like a reply: no HTML, no
// images, links opened only after the user confirms.

import { useMemo } from "react";

import { Markdown } from "@/components/Markdown";
import { readmeMarkdown } from "./logic";

export function Readme({ text, page }: { text: string; page: string }) {
  const source = useMemo(() => readmeMarkdown(text, page), [text, page]);
  return source ? <Markdown source={source} /> : null;
}
