// DeMARS version checks for OpenCode: the same three points the Codex hooks covered.
// session start -> report the version state; campaign-agent launch (task tool) -> block unless the
// version gate passes; session idle -> report a frozen-state violation. Never modifies files.
import { join } from "node:path"

export const DemarsVersion = async ({ $, worktree, client }) => {
  const check = join(worktree, "version", "check_version.py")

  const log = async (level, message) => {
    try {
      await client.app.log({ body: { service: "demars-version", level, message } })
    } catch {
      console.error(`[demars-version] ${message}`)
    }
  }

  return {
    event: async ({ event }) => {
      if (event.type === "session.created") {
        const r = await $`python3 ${check} session-start`.quiet().nothrow()
        await log("info", r.stdout.toString().trim())
      }
      if (event.type === "session.idle") {
        const r = await $`python3 ${check} stop < ${new Response("{}")}`.quiet().nothrow()
        if (r.exitCode !== 0) await log("error", r.stderr.toString().trim())
      }
    },
    "tool.execute.before": async (input, output) => {
      if (input.tool !== "task") return
      const hook = JSON.stringify({ tool_name: "task", tool_input: output.args })
      const r = await $`python3 ${check} gate < ${new Response(hook)}`.quiet().nothrow()
      if (r.exitCode !== 0) throw new Error(r.stderr.toString().trim())
    },
  }
}
