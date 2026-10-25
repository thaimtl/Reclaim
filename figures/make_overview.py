"""How Silent Voice works, for the README: figures/overview-{light,dark}.png.

    pip install playwright && playwright install chromium
    python figures/make_overview.py
"""
import base64
import os
import re

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1600, 750
BRAND = {'claude': '#D97757', 'twilio': '#F22F46', 'cartesia': '#309D4B', 'arduino': '#00878F',
         'mediapipe': '#0097A7', 'pytorch': '#EE4C2C', 'nvidia': '#76B900'}
CAMERA, EMG = '#2f6fde', '#c8612a'
THEMES = {
    'light': dict(bg='#ffffff', card='#ffffff', line='#d8dee4', ink='#1f2328', sub='#59636e', arrow='#8c959f',
                  tile='#f2f4f7', hero='#f4f8ff', hero_line='#a8c1ec'),
    'dark': dict(bg='#0d1117', card='#151b23', line='#3d444d', ink='#f0f6fc', sub='#9198a1', arrow='#6e7681',
                 tile='#232a33', hero='#121d30', hero_line='#35578f'),
}


def logo(name):
    s = open(os.path.join(HERE, 'logos', f'{name}.svg')).read()
    return re.search(r'viewBox="([^"]+)"', s).group(1), re.search(r' d="([^"]+)"', s).group(1)


def brand_mark(name, x, y, size, color):
    vb, d = logo(name)
    vw, vh = map(float, vb.split()[2:])
    k = size / max(vw, vh)
    return (f'<g transform="translate({x + (size - vw * k) / 2},{y + (size - vh * k) / 2}) scale({k})">'
            f'<path d="{d}" fill="{color}"/></g>')


ICONS = {
    'webcam': '<circle cx="12" cy="10" r="7"/><circle cx="12" cy="10" r="2.5"/><path d="M8 21h8M12 17v4"/>',
    'eye': '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
    'wave': '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    'phone': '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 '
             '1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 '
             '9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>',
    'lock': '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    'screen': '<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4M6 8h8M6 12h12"/>',
}


def head(x, y, size, t):
    """The pad figure: a head with the four EMG pad pairs, as in images/pad-placement.png."""
    k = size / 56
    dots = ''.join(f'<circle cx="{cx}" cy="{cy}" r="2.6" fill="{EMG}"/>'
                   for cx, cy in [(31, 24), (34, 23.5), (22, 30.5), (24.5, 32), (35, 31), (33.5, 33), (26, 35.5),
                                  (29.5, 35.5)])
    return (f'<g transform="translate({x},{y}) scale({k})" fill="none" stroke="{t["ink"]}" stroke-width="1.8">'
            f'<ellipse cx="28" cy="23" rx="12" ry="14.5"/><path d="M23 36.5v6M33 36.5v6M14 50c3-5 8-7.5 14-7.5s11 '
            f'2.5 14 7.5"/><path d="M24 28.5q4 2 8 0" stroke-width="1.4"/>{dots}</g>')


def card(t, x, y, w, h, title, lines, icon, badges=(), hero=False):
    fill, stroke = (t['hero'], t['hero_line']) if hero else (t['card'], t['line'])
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}" '
           f'stroke-width="1.5"/>']
    tx, ty, ts = x + 20, y + 22, 56
    kind, arg = icon
    if kind == 'brand' and arg == 'cartesia':
        out += [f'<rect x="{tx}" y="{ty}" width="{ts}" height="{ts}" rx="14" fill="{BRAND[arg]}"/>',
                brand_mark(arg, tx + 14, ty + 14, 28, '#ffffff')]
    elif kind == 'brand':
        out += [f'<rect x="{tx}" y="{ty}" width="{ts}" height="{ts}" rx="14" fill="{BRAND[arg]}" '
                f'fill-opacity="0.14"/>', brand_mark(arg, tx + 13, ty + 13, 30, BRAND[arg])]
    elif kind == 'head':
        out += [f'<rect x="{tx}" y="{ty}" width="{ts}" height="{ts}" rx="14" fill="{t["tile"]}"/>',
                head(tx, ty, ts, t)]
    else:
        color = arg or t['ink']
        tile = f'fill="{color}" fill-opacity="0.12"' if arg else f'fill="{t["tile"]}"'
        out += [f'<rect x="{tx}" y="{ty}" width="{ts}" height="{ts}" rx="14" {tile}/>',
                f'<g transform="translate({tx + 14},{ty + 14}) scale({28 / 24})" fill="none" stroke="{color}" '
                f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[kind]}</g>']
    for i, b in enumerate(badges):
        out.append(brand_mark(b, x + w - 40 - 30 * i, y + 18, 20, BRAND[b]))
    out.append(f'<text x="{x + 94}" y="{y + 47}" font-size="20" font-weight="700" fill="{t["ink"]}">{title}</text>')
    for i, line in enumerate(lines):
        out.append(f'<text x="{x + 94}" y="{y + 74 + 23 * i}" font-size="15.5" fill="{t["sub"]}">{line}</text>')
    return ''.join(out)


def arrow(t, d, label=None, at=None, anchor='middle'):
    out = f'<path d="{d}" fill="none" stroke="{t["arrow"]}" stroke-width="2" marker-end="url(#head)"/>'
    for i, line in enumerate(label or []):
        out += (f'<text x="{at[0]}" y="{at[1] + 19 * i}" font-size="14" text-anchor="{anchor}" '
                f'fill="{t["sub"]}">{line}</text>')
    return out


def meter(t, x, y, w):
    """One score built from the three readers, with the lock threshold marked."""
    bar, labels, at = [], [], x
    for share, color, name in [(0.38, CAMERA, 'camera'), (0.25, EMG, 'EMG'), (0.19, BRAND['claude'], 'Claude')]:
        bar.append(f'<rect x="{at}" y="{y}" width="{w * share}" height="10" fill="{color}"/>')
        labels.append(f'<text x="{at + w * share / 2}" y="{y + 30}" font-size="13" text-anchor="middle" '
                      f'fill="{t["sub"]}">{name}</text>')
        at += w * share
    lock = x + w * 0.7
    return (f'<clipPath id="bar"><rect x="{x}" y="{y}" width="{w}" height="10" rx="5"/></clipPath>'
            f'<rect x="{x}" y="{y}" width="{w}" height="10" rx="5" fill="{t["line"]}"/>'
            f'<g clip-path="url(#bar)">{"".join(bar)}</g>{"".join(labels)}'
            f'<path d="M{lock},{y - 7} V{y + 17}" stroke="{t["ink"]}" stroke-width="2"/>'
            f'<text x="{lock}" y="{y - 13}" font-size="13" text-anchor="middle" fill="{t["ink"]}">lock</text>')


def figure(t):
    c1, c2, c3, c4, cw, jw = 40, 430, 820, 1230, 330, 350
    r1, r2, r3, ch = 100, 290, 510, 110
    mid1, mid2, mid3 = r1 + ch / 2, r2 + ch / 2, r3 + ch / 2
    jy, jh = r1, r2 + ch - r1
    parts = [f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>']
    for x, w, name in [(c1, cw, 'Sense'), (c2, cw, 'Read'), (c3, jw, 'Decide'), (c4, cw, 'Speak')]:
        parts.append(f'<text x="{x + w / 2}" y="62" font-size="16" font-weight="700" text-anchor="middle" '
                     f'fill="{t["sub"]}" letter-spacing="0.4">{name}</text>')
    parts += [
        card(t, c1, r1, cw, ch, 'Webcam', ['Lips and facial expression'], ('webcam', CAMERA)),
        card(t, c1, r2, cw, ch, '4 EMG pads', ['Face muscle signals,', 'read by an Arduino'], ('head', None),
             ['arduino']),
        card(t, c1, r3, cw, ch, 'The judge', ['Talks normally on', 'a phone call'], ('phone', None)),
        card(t, c2, r1, cw, ch, 'Auto-AVSR', ['Reads the lips, again', 'every 100 ms'], ('eye', CAMERA),
             ['mediapipe']),
        card(t, c2, r2, cw, ch, 'SVR-1, our model', ['Reads the tongue and jaw', 'the camera can’t see'],
             ('wave', EMG), ['pytorch']),
        card(t, c2, r3, cw, ch, 'Cartesia Ink', ['Turns the judge’s speech', 'into text'], ('brand', 'cartesia')),
        card(t, c3, jy, jw, jh, 'Joint search', ['Weighs camera, EMG and Claude', 'into one confidence score.',
                                                  'A word locks once the score', 'clears a threshold that moves',
                                                  'with the conversation.'], ('lock', None), hero=True),
        meter(t, c3 + 94, jy + 218, jw - 124),
        f'<text x="{c3 + 94}" y="{jy + jh - 22}" font-size="15.5" fill="{t["sub"]}">Still unsure: '
        f'“Please repeat”</text>',
        card(t, c3, r3, jw, ch, 'Claude', ['Follows the whole call, predicts', 'words, fixes look-alikes'],
             ('brand', 'claude')),
        card(t, c4, r1, cw, ch, 'Live captions', ['Both sides of the call,', 'on the dashboard'], ('screen', None)),
        card(t, c4, r2, cw, ch, 'Cartesia Sonic', ['Austin’s cloned voice, with', 'the emotion on his face'],
             ('brand', 'cartesia')),
        card(t, c4, r3, cw, ch, 'Twilio call', ['The voice goes into the call', 'while he’s still mouthing'],
             ('brand', 'twilio')),
    ]
    gap = 6
    for y in (mid1, mid2, mid3):
        parts.append(arrow(t, f'M{c1 + cw + gap},{y} H{c2 - gap}'))
    for y in (mid1, mid2):
        parts += [arrow(t, f'M{c2 + cw + gap},{y} H{c3 - gap}'), arrow(t, f'M{c3 + jw + gap},{y} H{c4 - gap}')]
    parts += [
        arrow(t, f'M{c2 + cw + gap},{mid3} H{c3 - gap}'),
        arrow(t, f'M{c3 + 145},{r3 - gap} V{jy + jh + gap}', ['predictions,', 'repairs'], (c3 + 133, jy + jh + 52),
              'end'),
        arrow(t, f'M{c3 + 205},{jy + jh + gap} V{r3 - gap}', ['uncertain', 'words'], (c3 + 217, jy + jh + 52),
              'start'),
        arrow(t, f'M{c4 + cw / 2},{r2 + ch + gap} V{r3 - gap}'),
        arrow(t, f'M{c4 + cw / 2},{r3 + ch + gap} V{r3 + ch + 52} H{c1 + cw / 2} V{r3 + ch + gap}'),
        f'<rect x="{W / 2 - 170}" y="{r3 + ch + 38}" width="340" height="28" fill="{t["bg"]}"/>',
        f'<text x="{W / 2}" y="{r3 + ch + 57}" font-size="14" text-anchor="middle" fill="{t["sub"]}">'
        f'The judge hears Austin in his own voice</text>',
        brand_mark('nvidia', W / 2 - 330, H - 52, 22, BRAND['nvidia']),
        f'<text x="{W / 2 - 300}" y="{H - 35}" font-size="15" fill="{t["sub"]}">Runs live on one laptop with an '
        f'RTX 5060. Every message is logged and can be replayed.</text>',
    ]
    marker = (f'<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
              f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{t["arrow"]}"/></marker></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="Liberation Sans">{marker}{"".join(parts)}</svg>')


def font(style, weight):
    data = base64.b64encode(open(os.path.join(HERE, 'fonts', f'LiberationSans-{style}.ttf'), 'rb').read()).decode()
    return (f'@font-face{{font-family:"Liberation Sans";font-weight:{weight};'
            f'src:url(data:font/ttf;base64,{data})}}')


if __name__ == '__main__':
    css = font('Regular', 400) + font('Bold', 700) + 'body{margin:0}'
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': W, 'height': H}, device_scale_factor=2)
        for name, t in THEMES.items():
            page.set_content(f'<style>{css}</style>{figure(t)}')
            page.evaluate('document.fonts.ready')
            out = os.path.join(HERE, f'overview-{name}.png')
            page.locator('svg').first.screenshot(path=out)
            print('wrote', os.path.relpath(out))
        browser.close()
