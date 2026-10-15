import { isSafeUrl, parseInline, parseMarkdown, type Block } from "../markdown";

const types = (blocks: Block[]) => blocks.map((b) => b.type);

describe("blocks", () => {
  it("splits paragraphs, headings, rules, quotes, code and tables", () => {
    const src = [
      "# Title",
      "",
      "Para one",
      "still para",
      "",
      "---",
      "> quoted",
      "lazy",
      "",
      "```js",
      "const a = 1;",
      "```",
      "| a | b |",
      "|:--|--:|",
      "| 1 | 2 \\| 3 |",
    ].join("\n");
    const blocks = parseMarkdown(src);
    expect(types(blocks)).toEqual(["heading", "paragraph", "hr", "quote", "code", "table"]);
    expect(blocks[1]).toMatchObject({ text: "Para one\nstill para" });
    expect(blocks[3]).toMatchObject({ children: [{ type: "paragraph", text: "quoted\nlazy" }] });
    expect(blocks[4]).toMatchObject({ lang: "js", text: "const a = 1;" });
    expect(blocks[5]).toMatchObject({
      header: ["a", "b"],
      aligns: ["left", "right"],
      rows: [["1", "2 | 3"]],
    });
  });

  it("keeps an unclosed fence open to the end (streaming)", () => {
    const blocks = parseMarkdown("text\n```py\nprint(1)\n");
    expect(blocks[1]).toMatchObject({ type: "code", lang: "py", text: "print(1)\n" });
  });

  it("nests lists by indent and keeps ordered starts", () => {
    const blocks = parseMarkdown("3. three\n4. four\n   - sub a\n   - sub b\n\nafter");
    expect(blocks[0]).toMatchObject({ type: "list", ordered: true, start: 3 });
    const list = blocks[0] as Extract<Block, { type: "list" }>;
    expect(list.items).toHaveLength(2);
    expect(types(list.items[1]!)).toEqual(["paragraph", "list"]);
    expect(blocks[1]).toMatchObject({ type: "paragraph", text: "after" });
  });

  it("gives each block its raw source so unchanged blocks can be reused", () => {
    const a = parseMarkdown("one\n\ntwo");
    const b = parseMarkdown("one\n\ntwo and more");
    expect(a[0]!.raw).toBe(b[0]!.raw);
    expect(a[1]!.raw).not.toBe(b[1]!.raw);
  });

  it("drops paragraph lines' leading spaces, as the web UI's <p> does", () => {
    expect(parseMarkdown("  one\n   two")[0]).toMatchObject({ type: "paragraph", text: "one\ntwo" });
  });

  it("treats raw HTML as text", () => {
    expect(parseMarkdown("<script>alert(1)</script>")).toEqual([
      { type: "paragraph", text: "<script>alert(1)</script>", raw: "<script>alert(1)</script>" },
    ]);
  });
});

describe("inline", () => {
  it("parses emphasis, code and line breaks", () => {
    expect(parseInline("**b** *i* ~~s~~ `c`\nx")).toEqual([
      { type: "strong", children: [{ type: "text", text: "b" }] },
      { type: "text", text: " " },
      { type: "em", children: [{ type: "text", text: "i" }] },
      { type: "text", text: " " },
      { type: "del", children: [{ type: "text", text: "s" }] },
      { type: "text", text: " " },
      { type: "code", text: "c" },
      { type: "br" },
      { type: "text", text: "x" },
    ]);
  });

  it("leaves snake_case and lone markers alone", () => {
    expect(parseInline("a_b_c 2 * 3")).toEqual([{ type: "text", text: "a_b_c 2 * 3" }]);
  });

  it("makes links and autolinks", () => {
    expect(parseInline('[site](https://e.com "T") and https://x.org/a.')).toEqual([
      {
        type: "link",
        url: "https://e.com",
        title: "T",
        image: false,
        children: [{ type: "text", text: "site" }],
      },
      { type: "text", text: " and " },
      {
        type: "link",
        url: "https://x.org/a",
        title: "",
        image: false,
        children: [{ type: "text", text: "https://x.org/a" }],
      },
      { type: "text", text: "." },
    ]);
  });

  it("turns images into links and never into image nodes", () => {
    expect(parseInline("![cat](https://e.com/cat.png)")).toEqual([
      {
        type: "link",
        url: "https://e.com/cat.png",
        title: "",
        image: true,
        children: [{ type: "text", text: "Image: cat" }],
      },
    ]);
  });

  it("drops unsafe link schemes, keeping the label as text", () => {
    expect(parseInline("[click](javascript:alert(1)) [f](file:///etc/passwd)")).toEqual([
      { type: "text", text: "click" },
      { type: "text", text: " " },
      { type: "text", text: "f" },
    ]);
    expect(isSafeUrl("mailto:a@b.c")).toBe(true);
    expect(isSafeUrl("data:text/html,x")).toBe(false);
  });

  it("stays fast on pathological input", () => {
    const start = Date.now();
    parseInline("[".repeat(20000) + "*_~".repeat(5000));
    expect(Date.now() - start).toBeLessThan(2000);
  });
});
