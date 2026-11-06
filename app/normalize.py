"""Arabic normalisation — ARCH-02 §3.1.

ONE shared function, used identically at index time and query time, so
index-time and query-time normalisation can never drift apart.

The normalised text is stored ALONGSIDE the raw text in the Qdrant
payload. Citations and answers always display the RAW text; retrieval
uses the NORMALISED text. Normalisation must never leak into displayed
answers (e.g. changing ة to ه in a quoted price list would corrupt the
buyer's own wording — that is a defect, not a feature).
"""

import re
import unicodedata

# Arabic diacritics: U+064B–U+0652, U+0670
_DIACRITICS = re.compile(r"[\u064b-\u0652\u0670]")
_TATWEEL = "\u0640"
_ZERO_WIDTH_BIDI = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)
_WHITESPACE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Normalise Arabic text for retrieval (index time AND query time).

    Steps (ARCH-02 §3.1):
      strip diacritics | unify alef forms (أ/إ/آ/ٱ → ا) | ى → ي |
      ة → ه | remove tatweel | strip zero-width/bidi controls |
      normalise Arabic-Indic digits → 0-9 | collapse whitespace.
    """
    if text is None:
        return ""
    # Compatibility decomposition first: folds Arabic presentation forms
    # (U+FB50–U+FDFF, common in broken PDF text layers) to standard letters.
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_ALEF_VARIANTS)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace(_TATWEEL, "")
    text = _ZERO_WIDTH_BIDI.sub("", text)
    text = text.translate(_ARABIC_INDIC_DIGITS)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def tokenize_normalized(text: str) -> list:
    """Whitespace tokenisation over normalised text (for BM25)."""
    return normalize_arabic(text).split()
