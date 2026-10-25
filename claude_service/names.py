"""Words and names Austin says on every demo call: hints for the joint search from the first second, before Claude's
first prediction, and in every claude.predict after it. The one list both sides read (claude_service/brain.py adds it
to each prediction; fusion/core.py hints it always). Every word is in data/lexicon (cmudict.dict or
g2p_supplement.dict), so SVR-1 can grade it by sound.
"""
ALWAYS_WEIGHT = 0.4     # hint weight of an always-on word: a strong Claude prediction (~0.9) still counts more

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
