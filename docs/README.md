# Docs

| Doc | Read it for |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) ([PDF](ARCHITECTURE.pdf)) | How the whole system works, in plain words: the three readers, the real-time loop, the EMG model, the decoder, Claude, the voice |
| [PLAN.md](PLAN.md) ([PDF](PLAN.pdf)) | Who does what, the message formats between parts, the order of work, the recording session, training |
| [EVAN_WORKFLOW.md](EVAN_WORKFLOW.md) ([PDF](EVAN_WORKFLOW.pdf)) | Step-by-step plan for the real-time camera reader |
| [DECISIONS.md](DECISIONS.md) ([PDF](DECISIONS.pdf)) | What the team decided on October 24, and what each decision replaced |
| [recording/](recording/) | The sentences for the three recording sessions (`session1.txt`, `session2.txt`, `session3_test.txt`). The procedure is [RECORDING.md](../RECORDING.md) at the top of the repo |

The Markdown is the source of truth; the PDFs are copies for reading. After editing a doc, rebuild its PDF with `python docs/build_pdfs.py` (needs `pip install markdown playwright` and `playwright install chromium`; works offline, Mermaid is vendored in `vendor/`) and commit both.
