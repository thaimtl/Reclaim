# Reclaim

Silent speech for people who lost their voice box.
You mouth the words, and Reclaim says them out loud in your own voice, while you are still mouthing.

Built at Cal Hacks 13.0, October 23 to 25, 2026.

## Who it's for

A laryngectomy removes the voice box, usually after throat cancer.
People can still move their lips, tongue, and jaw, so the words are there without the sound.
Most rely on an electrolarynx, a handheld buzzer that sounds robotic and needs a free hand.

Our demo speaker is Austin, who is healthy; no patients take part in the demo.

## How it works

One mode: open mouth, no sound, real time.
Three readers work together, because each sees what the others can't.

```mermaid
flowchart LR
    W[Webcam] --> AV[Camera reader<br/>Auto-AVSR]
    P[6 EMG pads] --> E[EMG reader<br/>SVR-1]
    AV --> D[Fusion decoder]
    E --> D
    M[Partner's mic] --> I[Cartesia Ink]
    I --> C[Claude<br/>whole conversation]
    C <--> D
    D --> CAP[Live captions<br/>both speakers]
    I --> CAP
    D --> V[Cartesia Sonic<br/>cloned voice + emotion]
```

- The camera reads the lips and does most of the work. It also reads the facial expression, which sets the voice's emotion.
- The EMG model reads the face muscles, so it sees the tongue and jaw that no camera can. It settles words that look alike on the lips ("bat", "mat", "pat"), proposes its own guesses, and keeps going when the face is hard to see.
- Claude follows the whole conversation, both sides. It predicts likely words, and it repairs look-alike mistakes ("bills" → "pills" after a question about medicine). It only swaps look-alike words, never adds ideas, and says "Please repeat" when it isn't sure.
- Cartesia speaks each word in Austin's cloned voice about a second after he mouths it, without waiting for the end of the sentence.
- A deliberate jaw clench stops the voice instantly, so the speaker stays in control.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Plan and owners: [docs/PLAN.md](docs/PLAN.md). What changed and why: [docs/DECISIONS.md](docs/DECISIONS.md).

## What we demo

Two minutes, live, at our table.

1. A judge asks Austin a question out loud; their words appear in the captions.
2. Austin answers by mouthing. His words appear as he mouths and play in his cloned voice while he is still going.
3. He smiles or looks surprised, and the voice's emotion follows.
4. A look-alike word gets fixed live, and the screen shows what Claude changed and what the EMG settled.
5. A jaw clench stops the voice mid-sentence.

## Stack

| Part | Tool |
|---|---|
| Lip reading | [Auto-AVSR](https://github.com/mpc001/auto_avsr), as is, re-read every 200 ms for real time |
| Face, mouth crop, expression | MediaPipe FaceLandmarker |
| Muscle signals | BioAmp EXG Pill boards and an Arduino Uno, see [HARDWARE.md](HARDWARE.md) |
| EMG model | SVR-1, ours: PyTorch, pre-trained on Berkeley's silent speech data, fine-tuned on Austin |
| Context and repairs | Claude, through the Claude API |
| Voice clone and speech | Cartesia Sonic 3.6, Instant Voice Clone |
| Partner's speech to text | Cartesia Ink |
| Compute | One laptop, RTX 5060 |

## Known limitations

- This is a prototype, not a medical device.
- Lip reading gets about 1 in 5 words wrong even on normal speaking video, and silent mouthing is harder; the EMG model and Claude are there to close that gap.
- The EMG model is trained for one person (Austin), from about 75 minutes of recordings.
- The voice runs about a second behind the lips, so that words can settle and Claude can repair them before they are spoken.

## Built on

- [Digital Voicing of Silent Speech](https://aclanthology.org/2020.emnlp-main.445.pdf), Gaddy and Klein, UC Berkeley, 2020, for the electrode sites and pre-training data.
- [Auto-AVSR](https://github.com/mpc001/auto_avsr), Ma et al., 2023, for lip reading.
- [Lipflow](https://github.com/amywork777/lipflow), MIT license, for live mouth cropping with MediaPipe.

## Team

Thai (EMG hardware), Austin (EMG model and fusion, and the speaker in the demo), Frank (Cartesia voice), and Evan (real-time lip reading and facial emotion).
