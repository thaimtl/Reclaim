# Docs

| Doc | Read it for |
| --- | --- |
| [REALTIME_PIPELINE.md](REALTIME_PIPELINE.md) ([PDF](REALTIME_PIPELINE.pdf)) | How the whole real-time system works, in plain words: the three readers, re-reading every 100 ms, confidence locking, Claude, the voice and the call |
| [BUILD_PLAN.md](BUILD_PLAN.md) ([PDF](BUILD_PLAN.pdf)) | What is left before the demo: where the code stands, the decisions on the hard problems, the timeline, the cut line and the numbers to show |
| [SVR1_SPEC.md](SVR1_SPEC.md) ([PDF](SVR1_SPEC.pdf)) | The EMG model, SVR-1, as an engineering spec: inputs, layers, outputs, losses, training, and the changes from the current code. Figure: [svr1_architecture.png](../figures/svr1_architecture.png) |
| [TEAM_TASKS_AND_INTERFACES.md](TEAM_TASKS_AND_INTERFACES.md) ([PDF](TEAM_TASKS_AND_INTERFACES.pdf)) | Who builds what, the message formats between parts, the order of work, recording and training |
| [EVAN_REALTIME_LIP_READING.md](EVAN_REALTIME_LIP_READING.md) ([PDF](EVAN_REALTIME_LIP_READING.pdf)) | Evan's step-by-step plan for the real-time camera reader |
| [recording/](recording/) | The sentences for the three recording sessions. The procedure is [RECORDING.md](../RECORDING.md) at the top of the repo |

The Markdown is the source of truth; the PDFs are copies for reading. After editing a doc, rebuild its PDF with `python docs/build_pdfs.py` (needs `pip install markdown playwright` and `playwright install chromium`; works offline, Mermaid is vendored in `vendor/`) and commit both.
