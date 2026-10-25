"""Make Austin's Cartesia Instant Voice Clone from a recording, save its id, and render test lines to listen to.

    venv-call/bin/python -m call.clone_voice austin.wav --write          # clone, put the id in call/keys.env
    venv-call/bin/python -m call.clone_voice --test                      # render test lines with CARTESIA_VOICE_ID

Recording (Cartesia's guidance): one speaker, a quiet room without echo, 10 to 60 s (Sonic 3.6 learns from up to
60 s), spoken conversationally in the mood the voice should have. Austin's own voice, with his consent.
"""
import argparse
import os
import re
import wave

from cartesia import Cartesia

from call.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = os.path.join(HERE, 'keys.env')
TEST_LINES = [                      # short demo-style replies, in the voice's neutral, warm and worried registers
    "I'd love a coffee with oat milk, please.",
    'Yes, but they lost my bag.',
    'Oh, that is great news. Thank you so much!',
]
MAX_BYTES = 16 * 1024 * 1024


def check_clip(path):
    """Size and, for WAV, length and channels. Returns a list of warnings; raises on a clip Cartesia will refuse."""
    size = os.path.getsize(path)
    if size > MAX_BYTES:
        raise SystemExit(f'{path} is {size / 1e6:.1f} MB; Cartesia takes at most 16 MB. Trim it or save it as mp3.')
    warn = []
    if path.lower().endswith('.wav'):
        with wave.open(path) as w:
            secs = w.getnframes() / w.getframerate()
            if w.getnchannels() > 1:
                warn.append('stereo: fine, but make sure only Austin is speaking')
        if secs < 10:
            warn.append(f'only {secs:.1f} s: aim for 10 to 60 s')
        elif secs > 60:
            warn.append(f'{secs:.0f} s: Sonic 3.6 learns from the first 60 s only')
    return warn


def write_voice_id(voice_id):
    """Set CARTESIA_VOICE_ID in call/keys.env, keeping every other line as it is."""
    lines = open(KEYS, encoding='utf-8').read().splitlines() if os.path.exists(KEYS) else []
    out, done = [], False
    for line in lines:
        if re.match(r'\s*CARTESIA_VOICE_ID\s*=', line):
            out.append(f'CARTESIA_VOICE_ID={voice_id}')
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f'CARTESIA_VOICE_ID={voice_id}')
    with open(KEYS, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')


def render_tests(client, cfg, voice_id, out_dir):
    paths = []
    for i, line in enumerate(TEST_LINES, 1):
        audio = b''.join(client.tts.bytes(model_id=cfg.tts_model, transcript=line, voice={'id': voice_id},
                                          output_format={'container': 'wav', 'encoding': 'pcm_s16le',
                                                         'sample_rate': 44100}, language='en'))
        path = os.path.join(out_dir, f'clone_test_{i}.wav')
        with open(path, 'wb') as f:
            f.write(audio)
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('clip', nargs='?', help='Austin speaking, 10 to 60 s (wav, mp3, ogg, flac or webm; convert m4a first)')
    ap.add_argument('--name', default='Austin (Reclaim)')
    ap.add_argument('--write', action='store_true', help='save the new voice id as CARTESIA_VOICE_ID in call/keys.env')
    ap.add_argument('--test', action='store_true', help='render the test lines with the voice (after cloning, or the saved one)')
    ap.add_argument('--out', default=HERE, help='where the test wavs go')
    args = ap.parse_args()
    cfg = Config()
    if not cfg.cartesia_key:
        raise SystemExit('CARTESIA_API_KEY is missing from call/keys.env')
    client = Cartesia(api_key=cfg.cartesia_key)
    voice_id = cfg.voice_id
    if args.clip:
        for w in check_clip(args.clip):
            print('note:', w)
        with open(args.clip, 'rb') as f:
            voice = client.voices.clone(clip=f, name=args.name, language='en',
                                        description='Reclaim demo: Austin\'s own voice, cloned with his consent')
        voice_id = voice.id
        print('cloned:', voice.name, voice_id)
        if args.write:
            write_voice_id(voice_id)
            print('saved CARTESIA_VOICE_ID in call/keys.env; restart the call server to use it')
        args.test = True
    if args.test:
        if not voice_id:
            raise SystemExit('no voice: pass a clip, or set CARTESIA_VOICE_ID in call/keys.env')
        for p in render_tests(client, cfg, voice_id, args.out):
            print('listen:', p)


if __name__ == '__main__':
    main()
