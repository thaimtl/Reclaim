"""Character vocabulary and text normalization.

Matches Gaddy's TextTransform with the jiwer 2.2.1 he pinned: transliterate to ASCII,
delete every character of string.punctuation, lowercase. The model predicts one of
37 characters or the CTC blank.
"""
import string

from unidecode import unidecode

CHARS = string.ascii_lowercase + string.digits + ' '
BLANK = len(CHARS)  # 37
NUM_CLASSES = len(CHARS) + 1  # 38
SPACE = CHARS.index(' ')

_CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}
_PUNCTUATION = set(string.punctuation)


def clean_text(text):
    text = unidecode(text)
    text = ''.join(c for c in text if c not in _PUNCTUATION)
    return text.lower()


def text_to_ids(text):
    return [_CHAR_TO_ID[c] for c in clean_text(text)]


def ids_to_text(ids):
    return ''.join(CHARS[i] for i in ids)


_ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
         'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
_ORDINAL_ONES = ['zeroth', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
                 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth',
                 'eighteenth', 'nineteenth']
_ORDINAL_TENS = {20: 'twentieth', 30: 'thirtieth'}
_MONTHS = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
           'october', 'november', 'december'}


def _cardinal(n):
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + ('' if ones == 0 else ' ' + _ONES[ones])


def _ordinal(n):
    if n < 20:
        return _ORDINAL_ONES[n]
    if n in _ORDINAL_TENS:
        return _ORDINAL_TENS[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + ' ' + _ORDINAL_ONES[ones]


def _year(y):
    hi, lo = divmod(y, 100)
    if lo == 0:
        return _cardinal(hi) + ' hundred'
    if lo < 10:
        return _cardinal(hi) + ' oh ' + _ONES[lo]
    return _cardinal(hi) + ' ' + _cardinal(lo)


def spell_out_date_time(text):
    """Spell out a closed-vocab date/time phrase the way it is read aloud.

    Conventions (the dataset does not document how phrases were read):
        "06:44 PM"          -> "six forty four p m"
        "03:05 AM"          -> "three oh five a m"
        "11:00 AM"          -> "eleven o clock a m"
        "March 31 1893"     -> "march thirty first eighteen ninety three"
        "Tuesday August 11" -> "tuesday august eleventh"
        "January 02 1905"   -> "january second nineteen oh five"
    """
    out = []
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if ':' in tok:
            h, m = (int(p) for p in tok.split(':'))
            words = _cardinal(h)
            if m == 0:
                words += ' o clock'
            elif m < 10:
                words += ' oh ' + _ONES[m]
            else:
                words += ' ' + _cardinal(m)
            out.append(words)
        elif tok.upper() in ('AM', 'PM'):
            out.append(' '.join(tok.lower()))
        elif tok.isdigit() and len(tok) == 4:
            out.append(_year(int(tok)))
        elif tok.isdigit() and i > 0 and tokens[i - 1].lower() in _MONTHS:
            out.append(_ordinal(int(tok)))
        elif tok.isdigit():
            out.append(_cardinal(int(tok)))
        else:
            out.append(tok.lower())
    return clean_text(' '.join(out))
