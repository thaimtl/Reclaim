"""Render the docs' Markdown to PDF, with Mermaid diagrams drawn.

    pip install markdown playwright && playwright install chromium
    python docs/build_pdfs.py            # all docs listed in DOCS
    python docs/build_pdfs.py TEAM_TASKS_AND_INTERFACES.md    # one doc

Markdown -> HTML (python-markdown), Mermaid blocks drawn by mermaid.js in headless Chromium, printed to Letter PDF.
Mermaid is vendored in docs/vendor/ (11.4.1, MIT), so the build works offline.
Links to the other docs in DOCS point to their PDFs next to this one; every other relative link points to the file
on GitHub, so links still work in a PDF opened on its own.
"""
import base64
import html
import os
import re
import sys

import markdown
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DOCS = ['REALTIME_PIPELINE.md', 'BUILD_PLAN.md', 'SVR1_SPEC.md', 'TEAM_TASKS_AND_INTERFACES.md', 'EVAN_REALTIME_LIP_READING.md']
MERMAID = os.path.join(HERE, 'vendor', 'mermaid.min.js')
MERMAID_JS = ''
REPO_URL = 'https://github.com/thaimtl/Reclaim/blob/main/'

CSS = """
@page { size: Letter; margin: 18mm 16mm; }
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.5;
       color: #1d1d1f; }
h1 { font-size: 21pt; margin: 0 0 8pt; }
h2 { font-size: 14.5pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ddd; padding-bottom: 3pt; }
h3 { font-size: 12pt; margin: 14pt 0 4pt; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #d0d0d0; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #f3f4f6; }
code { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 8.8pt; background: #f3f4f6; padding: 0 2pt;
       border-radius: 3px; }
pre { background: #f6f8fa; padding: 8pt; border-radius: 6px; overflow-x: hidden; white-space: pre-wrap;
      font-size: 8.6pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
pre.mermaid { background: none; text-align: center; page-break-inside: avoid; }
img { max-width: 100%; }
pre.mermaid svg { max-width: 100%; max-height: 165mm; width: auto; height: auto; }
a { color: #0b57d0; text-decoration: none; }
"""


def fix_link(m):
    """Sibling docs -> their PDFs; any other relative path -> the file on GitHub (main branch)."""
    target, anchor = m.group(1), m.group(2) or ''
    if target in DOCS:
        return f'href="{target[:-3]}.pdf"'
    path = os.path.relpath(os.path.normpath(os.path.join(HERE, target)), REPO)
    return f'href="{REPO_URL}{path}{anchor}"'


def embed_image(m):
    """Relative image -> data URI, since the page has no file base URL."""
    path = os.path.normpath(os.path.join(HERE, m.group(1)))
    mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml'}[os.path.splitext(path)[1].lower()]
    return f'src="data:{mime};base64,{base64.b64encode(open(path, "rb").read()).decode()}"'


def to_html(md_text):
    body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'sane_lists'])
    # fenced ```mermaid blocks -> <pre class="mermaid"> for mermaid.js
    body = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
                  lambda m: '<pre class="mermaid">' + html.unescape(m.group(1)) + '</pre>', body, flags=re.S)
    body = re.sub(r'href="([^"#:]+)(#[^"]*)?"', fix_link, body)
    body = re.sub(r'src="([^"#:]+)"', embed_image, body)
    return ('<html><head><meta charset="utf-8"><style>' + CSS + '</style>'
            '<script>' + MERMAID_JS + '</script></head><body>' + body +
            '<script>mermaid.initialize({startOnLoad: false, theme: "neutral", flowchart: {useMaxWidth: true}});'
            'mermaid.run().then(() => document.body.setAttribute("data-done", "1"))'
            '.catch(e => { document.body.setAttribute("data-error", String(e)); });</script></body></html>')


def main():
    global MERMAID_JS
    MERMAID_JS = open(MERMAID, encoding='utf-8').read()
    names = sys.argv[1:] or DOCS
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name in names:
            src = os.path.join(HERE, name)
            page = browser.new_page()
            page.set_content(to_html(open(src, encoding='utf-8').read()), wait_until='load')
            page.wait_for_function('document.body.hasAttribute("data-done") || document.body.hasAttribute("data-error")',
                                   timeout=60000)
            err = page.evaluate('document.body.getAttribute("data-error")')
            if err:
                raise SystemExit(f'{name}: mermaid failed: {err}')
            out = os.path.splitext(src)[0] + '.pdf'
            page.pdf(path=out, format='Letter', print_background=True,
                     margin={'top': '18mm', 'bottom': '18mm', 'left': '16mm', 'right': '16mm'})
            page.close()
            print('wrote', os.path.relpath(out, os.path.dirname(HERE)))
        browser.close()


if __name__ == '__main__':
    main()
