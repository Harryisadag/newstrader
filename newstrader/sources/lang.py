"""Which language a headline is in - a small, dependency-free guess.

The local ML engine (FinBERT) only understands English, so non-English text is skipped there instead of being
misread (the Claude engine reads any language). Non-Latin scripts are recognised by their letters; Latin-script
languages by common short words. When in doubt the answer is "en", because a wrong "not English" would hide news.
"""

from __future__ import annotations

import re
import unicodedata

# Unicode script ranges -> language (the most likely one for news feeds)
_SCRIPTS: list[tuple[str, str]] = [
    (r"[぀-ヿ]", "ja"),  # hiragana / katakana (checked before Chinese characters, which Japanese uses too)
    (r"[가-힯]", "ko"),  # hangul
    (r"[一-鿿]", "zh"),
    (r"[؀-ۿ]", "ar"),
    (r"[Ѐ-ӿ]", "ru"),
    (r"[ऀ-ॿ]", "hi"),
    (r"[฀-๿]", "th"),
    (r"[֐-׿]", "he"),
    (r"[Ͱ-Ͽ]", "el"),
]
_SCRIPT_RES = [(re.compile(p), code) for p, code in _SCRIPTS]

# Short, very common words that rarely appear in English text
_STOPWORDS: dict[str, set[str]] = {
    "en": {"the", "and", "of", "to", "in", "is", "for", "on", "with", "as", "at", "by", "from", "after", "its",
           "says", "will", "are", "was", "has", "new", "over", "into", "amid", "than", "this", "that"},
    "es": {"el", "la", "los", "las", "de", "del", "y", "que", "en", "por", "para", "con", "una", "un", "se", "su",
           "al", "es", "más", "tras", "sobre", "como"},
    "fr": {"le", "la", "les", "des", "du", "de", "et", "un", "une", "est", "dans", "pour", "sur", "au", "aux", "avec",
           "qui", "pas", "plus", "après", "selon", "ce"},
    "de": {"der", "die", "das", "und", "den", "dem", "des", "ist", "mit", "für", "von", "auf", "ein", "eine", "nicht",
           "im", "zu", "nach", "bei", "wie", "über", "sich", "auch"},
    "it": {"il", "lo", "gli", "della", "delle", "dei", "che", "per", "con", "una", "un", "sono", "nel", "alla",
           "dopo", "anche", "più", "tra", "di", "del", "sul", "sull", "nella", "non"},
    "pt": {"o", "os", "da", "do", "das", "dos", "que", "em", "para", "com", "uma", "um", "não", "mais", "como",
           "na", "no", "ao", "após", "sobre"},
    "nl": {"de", "het", "een", "en", "van", "op", "voor", "met", "niet", "zijn", "naar", "bij", "ook", "dat"},
    "tr": {"ve", "bir", "bu", "ile", "için", "da", "de", "olarak", "daha", "çok", "sonra", "gibi"},
}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str) -> str:
    """"en", another language code, or "" when there is too little text to tell."""
    text = (text or "")[:2000]
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 8:
        return ""
    han = len(_SCRIPT_RES[2][0].findall(text))
    kana = len(_SCRIPT_RES[0][0].findall(text))
    if kana >= 2 and (kana + han) / len(letters) > 0.3:
        return "ja"  # Japanese mixes Chinese characters with kana
    for rx, code in _SCRIPT_RES:
        n = len(rx.findall(text))
        if n >= 4 and n / len(letters) > 0.3:
            return code
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return ""
    scores = {lang: sum(1 for w in words if w in sw) for lang, sw in _STOPWORDS.items()}
    # letters English never uses are a strong hint (ñ, ç, ß, ã, õ, ë... - but not é, common in borrowed names)
    accents = sum(1 for ch in text if unicodedata.category(ch) == "Ll" and ch in "ñçßãõäöüàèìòùâêîôûœ¿¡ğışı")
    best = max(scores, key=lambda k: scores[k])
    if best == "en" or scores[best] < 2:
        return "en" if scores["en"] or accents < 2 else ""
    # another language must clearly beat English (company names and English loanwords are everywhere)
    if scores[best] >= max(2, scores["en"] * 2) or (accents >= 2 and scores[best] > scores["en"]):
        return best
    return "en"
