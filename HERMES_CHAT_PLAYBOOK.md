# Hermes Chat Playbook — for Devin sessions

How to start a Hermes chat from a Devin session and keep it alive across turns.
Proven 2026-08-26 on this machine (Hermes v0.20.1, Windows/PowerShell, exec tool).

## 1. Prereqs

- `hermes` CLI on PATH (`$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\hermes.exe`)
- Sanity check: `hermes --version`, `hermes chat --help`

## 2. Start a new chat (one-shot)

```powershell
$q = @'
<Intro + context + numbered questions>
'@
hermes chat -q $q --source devin --in "<github-root>\Argos" -Q
```

- Capture the `session_id: ...` line printed at the end of the output — it is the thread handle.
- `--source devin` tags the session so it doesn't clutter the user's session list (the flag exists for third-party integrations).
- `--in DIR` lets Hermes read the workspace; resumed sessions restore their recorded cwd.
- `-Q` = clean programmatic output (no banner/spinner). Drop `-Q` to see tool previews.

## 3. Maintain continuity (every follow-up)

```powershell
$q = @'
<Follow-up message>
'@
hermes chat -q $q --resume <SESSION_ID> --source devin -Q
```

- Use the explicit session ID captured in step 2. `--resume latest` works but is ambiguous when other sessions exist.
- Each call is fire-and-forget (no PTY, no babysitting), but the thread extends across calls: same session, same state — Hermes keeps its task list and context across resumption and compression.
- On resume the output confirms the thread: `Resumed session <id> "<title>" (N user messages, M total)`.

## 4. PowerShell quoting — critical

- Always pass the message via a single-quoted here-string: `@'` ... `'@`.
- Do NOT use double-quote characters (`"`) anywhere in the message text. Windows argv parsing splits arguments on them and the command dies with `unrecognized arguments: <mid-message text>`. Rephrase instead.
- Apostrophes inside the here-string are fine. `$` is literal in `@'...'@` (no interpolation).

## 5. Timing and errors

- `hermes chat` can take minutes (LLM + tool calls). Give the exec a long timeout; if it backgrounds, poll `get_output` until the final answer + `session_id:` line appear.
- Long output is truncated to an overflow file — read it.
- `session storage was busy (another Hermes process was writing to the state database)` → another Hermes is running (e.g. the user's own session). Wait 20–60s and resend the same message; it is saved.
- No TTY needed. If a hook-approval prompt ever blocks non-interactive mode, add `--accept-hooks`.

## 6. Getting good answers (proven in the Argos review exchange)

- First message: introduce yourself, name the repo, state what you want, ask numbered questions.
- Reference prior replies explicitly: "in your last reply you said X".
- Challenge claims Hermes asserts from memory — it corrects itself when shown repo greps or artifacts (it retracted two claims in one exchange: a 449/500 headline and an LOC "fix"). Its own house rule: verify in code, not handoffs.
- Ask for evidence (file paths, counts, arithmetic) rather than opinions; it will go measure.

## 7. Session hygiene

- New thread = new one-shot without `--resume` (keep `--source devin`).
- Nothing to kill: threads are just session records; sessions are listed/tagged via `hermes sessions`.
