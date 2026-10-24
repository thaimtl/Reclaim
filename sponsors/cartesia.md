# Cartesia

Source: https://docs.cartesia.ai/calhacks, read October 23, 2026.
Prize: Meta glasses for the winning team, up to 4 people.

## How they judge, 25% each

| Criterion | What they want | Our answer |
|---|---|---|
| Problem and originality | Useful, clearly scoped, thoughtfully designed | Giving people back their voice after a laryngectomy |
| Use of Cartesia | Core to the experience, responsive, natural voice | Without the voice there is no product |
| Execution | Main flow works, edge cases handled | Speaker confirms before playback, "Please repeat" when unsure |
| Impact and responsibility | Accessibility, privacy, responsible use | Clone only your own voice, with consent |

## What to emphasize

- Voice is the product, not a feature added on top.
The user speaks only through Cartesia.
- Their own voice, not a robot.
Instant Voice Clone means a laryngectomee sounds like themselves again, unlike an electrolarynx.
- Say out loud what each Cartesia part does: Sonic speaks, Instant Voice Clone makes it their voice, Ink hears the partner's question.
- Speed matters in a conversation.
Stream Claude's fixed text into the Sonic WebSocket so speech starts before Claude finishes.
- Show Cartesia's part and Claude's part as separate steps in the demo, since other tools are allowed but Cartesia's share must be obvious.

## Responsibility points to say out loud

- A voice clone is someone's identity.
We clone only the user's own voice, from their own recordings, with their consent.
- Nothing plays without the speaker confirming the sentence first.
- Cartesia API keys stay on the server.
If the dashboard calls Cartesia from the browser, mint a short-lived token on the server.

## Build rules from their page

- Use Sonic 3.6.
- Set speed, volume, and emotion through `generation_config`, not the old top-level `speed` field.
- Don't use the Line SDK.
Cartesia stops hosting it on December 1, 2026.

## Submission checklist

- Working demo.
- Two-minute pitch, demo first: one full user flow, then the problem, then how we built it.
- README with setup steps, the Cartesia features we used, and known limitations.
Our README has no setup steps yet.
- Judges can read the source code.
- No API keys committed.
