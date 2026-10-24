# Reclaim

Cal Hacks 13.0 project, October 23 to 25, 2026: silent speech for people who lost their voice box.
Read README.md and HARDWARE.md before answering.

Decisions already made, so don't re-litigate them without new evidence:
- Free speech is the main goal and must work no matter what. Quick phrases is the stretch goal.
- Free speech: webcam, Auto-AVSR lip reading, Claude fixing mistakes, Cartesia voice. No training.
- Quick phrases: 6 BioAmp EXG Pill channels into an Arduino Uno, our classifier for 10 phrases plus rest, then a Jev context check.
- Claude stays the step that fixes lip reading, since we're in the Anthropic track.
- No pad on the throat, no AD8232, no ESP32 in the rig, and each board has its own REF pad.
- Two laptops: Austin's runs free speech and the dashboard, plugged in. Thai's Mac is the only machine wired to the pads, on battery. They connect over a hotspot, never a cable.

Never commit API keys or sponsor codes.
Diagrams are Mermaid in Markdown.
Read `sponsors/*.md` before any pitch, demo, or sponsor-facing work.
Every change goes through a PR. Pushing straight to main is blocked.
