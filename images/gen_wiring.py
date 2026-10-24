# Usage: python3 gen_wiring.py out.svg, then screenshot it at 2x in headless Chrome with light mode to make wiring.png.
import sys

W, H = 1200, 720
FONT = '-apple-system, &quot;Segoe UI&quot;, Helvetica, Arial, sans-serif'
P = 19
Y0 = 100
out = []
a = out.append

a(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Wiring diagram on a 400-hole breadboard: the Arduino Uno's 5V goes to the left plus column and GND to the left minus column, each of the 4 BioAmp boards takes VCC from the left plus column and GND from the left minus column, and each board's OUT pin goes to one of the Uno's A0 to A3 pins. The right rails and the middle rows are unused.">
  <style>
    .bg {{ fill: #FFFFFF; }}
    .part {{ fill: #F3F5F4; stroke: #1B2320; stroke-width: 1.6; }}
    .bb {{ fill: #FAFAFA; stroke: #8A9792; stroke-width: 1.2; }}
    .groove {{ fill: #E6E9E8; }}
    .pin {{ fill: #1B2320; }}
    .pin-off {{ fill: #A9B4B0; }}
    .hole {{ fill: #9AA5A1; }}
    .unused {{ opacity: 0.4; }}
    .rail-p {{ stroke: #D1242F; stroke-width: 2; }}
    .rail-n {{ stroke: #0969DA; stroke-width: 2; }}
    .w {{ fill: none; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
    .halo {{ stroke: #FFFFFF; stroke-width: 9; }}
    .w-5v {{ stroke: #D1242F; }}
    .w-gnd {{ stroke: #1B2320; }}
    .w-sig {{ stroke: #B4521C; }}
    .dot-5v {{ fill: #D1242F; }}
    .dot-gnd {{ fill: #1B2320; }}
    .cable {{ fill: none; stroke: #5B6A65; stroke-width: 2; stroke-dasharray: 5 4; }}
    .t {{ fill: #1B2320; font: 600 13px {FONT}; }}
    .h {{ fill: #1B2320; font: 600 16px {FONT}; }}
    .lbl {{ fill: #1B2320; font: 500 11px {FONT}; }}
    .lbl-off {{ fill: #8A9792; font: 400 11px {FONT}; }}
    .sub {{ fill: #5B6A65; font: 400 12px {FONT}; }}
    .col {{ fill: #5B6A65; font: 400 11px {FONT}; }}
    .p {{ fill: #D1242F; font: 700 15px {FONT}; }}
    .n {{ fill: #0969DA; font: 700 15px {FONT}; }}
    @media (prefers-color-scheme: dark) {{
      .bg {{ fill: #0D1117; }}
      .part {{ fill: #161B22; stroke: #E3EAE7; }}
      .bb {{ fill: #1C2128; stroke: #6E7B76; }}
      .groove {{ fill: #0D1117; }}
      .pin {{ fill: #E3EAE7; }}
      .pin-off, .hole {{ fill: #4A5550; }}
      .halo {{ stroke: #0D1117; }}
      .rail-p, .w-5v {{ stroke: #FF7B72; }}
      .rail-n {{ stroke: #58A6FF; }}
      .w-gnd {{ stroke: #E3EAE7; }}
      .w-sig {{ stroke: #E58A52; }}
      .dot-5v {{ fill: #FF7B72; }}
      .dot-gnd {{ fill: #E3EAE7; }}
      .cable {{ stroke: #97A5A0; }}
      .t, .h, .lbl {{ fill: #E3EAE7; }}
      .lbl-off {{ fill: #6E7B76; }}
      .sub, .col {{ fill: #97A5A0; }}
      .p {{ fill: #FF7B72; }}
      .n {{ fill: #58A6FF; }}
    }}
  </style>
  <rect class="bg" width="{W}" height="{H}"/>''')

# Rail holes come in 5 groups of 5, with a one-hole gap between groups
rail_rows = [Y0 + g * 6 * P + h * P for g in range(5) for h in range(5)]
main_rows = [Y0 + r * P for r in range(30)]

# Breadboard, laid out like the photo: rails, a-e, groove, f-j, rails
LP, LN = 410, 429
RP, RN = 702, 721
MAIN_A = [474 + i * P for i in range(5)]
MAIN_F = [580 + i * P for i in range(5)]
BT, BB = 62, 676

def hole(x, y):
    a(f'  <rect class="hole" x="{x-2.5}" y="{y-2.5}" width="5" height="5"/>')

def rail(x0, xp, xn, extra=""):
    a(f'  <g{extra}>')
    a(f'  <rect class="bb" x="{x0}" y="{BT}" width="66" height="{BB-BT}" rx="4"/>')
    a(f'  <line class="rail-p" x1="{xp-12}" y1="{Y0-8}" x2="{xp-12}" y2="{rail_rows[-1]+8}"/>')
    a(f'  <line class="rail-n" x1="{xn+12}" y1="{Y0-8}" x2="{xn+12}" y2="{rail_rows[-1]+8}"/>')
    for yy in (BT + 20, BB - 8):
        a(f'  <text class="p" x="{xp-12}" y="{yy}" text-anchor="middle">+</text>')
        a(f'  <text class="n" x="{xn+12}" y="{yy}" text-anchor="middle">−</text>')
    for y in rail_rows:
        hole(xp, y)
        hole(xn, y)
    a('  </g>')

rail(386, LP, LN)
a(f'  <rect class="bb" x="458" y="{BT}" width="214" height="{BB-BT}" rx="4"/>')
a(f'  <rect class="groove" x="560" y="{BT+4}" width="10" height="{BB-BT-8}"/>')
for xs, letters in ((MAIN_A, "abcde"), (MAIN_F, "fghij")):
    for x, c in zip(xs, letters):
        a(f'  <text class="col" x="{x}" y="{BT+18}" text-anchor="middle">{c}</text>')
        a(f'  <text class="col" x="{x}" y="{BB-6}" text-anchor="middle">{c}</text>')
        for y in main_rows:
            hole(x, y)
rail(678, RP, RN, ' class="unused"')

a(f'  <text class="h" x="565" y="34" text-anchor="middle">Breadboard</text>')
a(f'  <text class="sub" x="419" y="52" text-anchor="middle">used</text>')
a(f'  <text class="sub" x="565" y="52" text-anchor="middle">rows unused</text>')
a(f'  <text class="sub" x="711" y="52" text-anchor="middle">unused</text>')

# Arduino Uno, turned so its power and analog header faces right and USB faces down
UX, UY, UW, UH = 40, 60, 220, 400
PX = UX + UW - 6
a(f'  <rect class="part" x="{UX}" y="{UY}" width="{UW}" height="{UH}" rx="10"/>')
a(f'  <rect class="part" x="{UX+70}" y="{UY+UH-4}" width="60" height="34" rx="3"/>')
a(f'  <text class="h" x="{UX+20}" y="{UY+32}">Arduino Uno</text>')
a(f'  <text class="sub" x="{UX+20}" y="{UY+52}">header edge facing right</text>')
a(f'  <text class="sub" x="{UX+100}" y="{UY+UH+50}" text-anchor="middle">USB-B to laptop</text>')

# 4 BioAmp boards, top to bottom A3..A0 so no two signal wires cross.
# Each board's VCC and GND line up with rail holes in one group.
sites = {
    "A3": "left cheek, above the mouth",
    "A2": "right cheek, below the mouth",
    "A1": "left corner of the chin",
    "A0": "under the chin",
}
KX, KW, KH = 810, 230, 96
boards = []
for i, ch in enumerate(["A3", "A2", "A1", "A0"]):
    pv = rail_rows[i * 5 + 1]
    boards.append((ch, pv, pv + P, pv + 2 * P))

pins = ["A5", "A4", "A3", "A2", "A1", "A0", None, "VIN", "GND", "GND", "5V", "3.3V", "RESET", "IOREF"]
used = {"A0", "A1", "A2", "A3", "5V"}
py = {}
y = boards[0][3] - 2 * 24
gnd_seen = False
for name in pins:
    if name is None:
        y += 16
        continue
    on = name in used or (name == "GND" and gnd_seen)
    if name == "GND":
        if gnd_seen:
            py["GND"] = y
        gnd_seen = True
    else:
        py[name] = y
    a(f'  <rect class="{"pin" if on else "pin-off"}" x="{PX-5}" y="{y-5}" width="10" height="10"/>')
    a(f'  <text class="{"lbl" if on else "lbl-off"}" x="{PX-12}" y="{y+4}" text-anchor="end">{name}</text>')
    y += 24

def wire(cls, pts):
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(pts))
    a(f'  <path class="w halo" d="{d}"/>')
    a(f'  <path class="w {cls}" d="{d}"/>')

def end(cls, x, y):
    a(f'  <circle class="dot-{cls}" cx="{x}" cy="{y}" r="4.5"/>')

# Uno power into the last rail group, run along the bottom so it crosses no signal wire
g5v, ggnd = rail_rows[22], rail_rows[21]
wire("w-5v", [(PX, py["5V"]), (278, py["5V"]), (278, g5v), (LP, g5v)])
end("5v", LP, g5v)
wire("w-gnd", [(PX, py["GND"]), (294, py["GND"]), (294, ggnd), (LN, ggnd)])
end("gnd", LN, ggnd)

cols = {"A2": 376, "A1": 356, "A0": 336}
for ch, pv, pg, po in boards:
    by = pv - 46
    a(f'  <rect class="part" x="{KX}" y="{by}" width="{KW}" height="{KH}" rx="6"/>')
    a(f'  <text class="t" x="{KX+14}" y="{by+18}">BioAmp → {ch}</text>')
    a(f'  <text class="sub" x="{KX+14}" y="{by+33}">{sites[ch]}</text>')
    for name, yy in (("VCC", pv), ("GND", pg), ("OUT", po)):
        a(f'  <rect class="pin" x="{KX-9}" y="{yy-4}" width="9" height="8"/>')
        a(f'  <text class="lbl" x="{KX+8}" y="{yy+4}">{name}</text>')
    a(f'  <rect class="pin" x="{KX+KW}" y="{by+36}" width="12" height="24" rx="2"/>')
    a(f'  <path class="cable" d="M{KX+KW+12},{by+48} C{KX+KW+40},{by+28} {KX+KW+60},{by+68} {KX+KW+88},{by+48}"/>')
    a(f'  <text class="sub" x="{KX+KW+94}" y="{by+44}">to</text>')
    a(f'  <text class="sub" x="{KX+KW+94}" y="{by+59}">pads</text>')
    wire("w-5v", [(KX - 9, pv), (LP, pv)])
    end("5v", LP, pv)
    wire("w-gnd", [(KX - 9, pg), (LN, pg)])
    end("gnd", LN, pg)
    if ch == "A3":
        wire("w-sig", [(KX - 9, po), (PX, po)])
    else:
        c = cols[ch]
        wire("w-sig", [(KX - 9, po), (c, po), (c, py[ch]), (PX, py[ch])])

# Legend
ly = 600
a(f'  <line class="w w-5v" x1="40" y1="{ly}" x2="70" y2="{ly}"/><text class="sub" x="78" y="{ly+4}">5V / VCC</text>')
a(f'  <line class="w w-gnd" x1="40" y1="{ly+24}" x2="70" y2="{ly+24}"/><text class="sub" x="78" y="{ly+28}">GND</text>')
a(f'  <line class="w w-sig" x1="40" y1="{ly+48}" x2="70" y2="{ly+48}"/><text class="sub" x="78" y="{ly+52}">OUT signal</text>')
a(f'  <text class="sub" x="{KX}" y="{H-14}">Match pins by the label printed on each board.</text>')
a('</svg>')

open(sys.argv[1], "w").write("\n".join(out) + "\n")
