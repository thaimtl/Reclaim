"""Words and names Austin says on every demo call: in the joint search from the first second, before Claude's first
prediction, and in every claude.predict after it. The one list both sides read (claude_service/brain.py adds it to
each prediction; fusion/core.py adds it to Claude's words always). At ALWAYS_WEIGHT they are weak predictions, so
they carry over the camera's reading only on a close lip match (fusion.search.WEAK_LIP_MIN), not in free speech that
merely sounds a bit like them. Every word is in data/lexicon (cmudict.dict or g2p_supplement.dict).
"""
ALWAYS_WEIGHT = 0.4     # an always-on word's weight: a strong Claude prediction (~0.9) counts more

ALWAYS = (
    'austin', 'frank', 'cal hacks', 'calhacks', 'mcgill university', 'red bull', 'meta', 'anthropic', 'cartesia',
    'coffee', 'asl', 'als', 'patients', 'laryngectomy', 'voice box', 'voice',
)


def always_words():
    """{word: ALWAYS_WEIGHT} for every word of every entry ('mcgill university' gives 'mcgill' and 'university')."""
    return {w: ALWAYS_WEIGHT for entry in ALWAYS for w in entry.split()}


def always_phrases():
    """The entries of more than one word, as phrases."""
    return [entry for entry in ALWAYS if ' ' in entry]
