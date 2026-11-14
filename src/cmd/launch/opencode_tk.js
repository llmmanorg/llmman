import { spawn } from "node:child_process"

const LLMMAN = __LLMMAN__

const compact = (input) =>
  new Promise((resolve) => {
    const child = spawn(LLMMAN, ["tk", "compact"], { timeout: 10_000 })
    let out = ""
    child.stdout.on("data", (chunk) => (out += chunk))
    child.on("error", () => resolve(""))
    child.on("close", (code) => resolve(code === 0 ? out : ""))
    child.stdin.on("error", () => {})
    child.stdin.end(input)
  })

export default {
  id: "llmman.tk",
  async setup(ctx) {
    await ctx.tool.hook("execute.after", async (event) => {
      const done = event.status === "completed" ? event.result.output : undefined
      if (event.tool !== "shell" || typeof done?.output !== "string") return
      const text = await compact(
        JSON.stringify({ command: event.input.command, output: done.output, exitCode: done.exitCode }),
      )
      if (!text) return
      event.result = {
        ...event.result,
        output: { ...done, output: text },
        content: [{ type: "text", text }],
      }
    })
  },
}
