# EMG model docs (Austin)

Planning docs for the EMG side: SVR-1, our silent-speech EMG model, and how it could fuse with the camera pipeline. Snapshot of October 23, 2026, exported from the team's Claude docs.

| File | What it is |
| --- | --- |
| [fused-emg-plan.md](fused-emg-plan.md) ([PDF](fused-emg-plan.pdf)) | Plan and status: how it works, results so far, real-time pipeline, benchmarks, data to record, next steps, risks |
| [svr1-explained.md](svr1-explained.md) ([PDF](svr1-explained.pdf)) | Plain-language walkthrough of the SVR-1 architecture figure, part by part, and what makes it real time |
| [svr1_architecture.png](svr1_architecture.png) | The architecture figure both docs refer to |

Where these docs differ from the settled decisions in `CLAUDE.md` (those decisions win until the team changes them):

- The docs use Gaddy electrode site 4 on the throat; `CLAUDE.md` says no pad on the throat.
- The docs fuse EMG into free speech and use a local ~4B LLM to rescore; `CLAUDE.md` keeps free speech camera-only with Claude fixing mistakes.
- The docs' fixed-sentence head is the same idea as the quick-phrases classifier (10 phrases plus rest).
