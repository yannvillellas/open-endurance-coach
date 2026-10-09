"""Character-level sanitisation for untrusted text.

Control characters, unpaired surrogates and Unicode layout/direction controls
can corrupt terminal output, forge log lines or crash outbound UTF-8 encoding.
Untrusted values are cleaned at the boundaries where they are rendered, logged
or sent, without rejecting the hub's data: dangerous characters are replaced
with U+FFFD so the operator can see that something was removed.
"""

import unicodedata

# Strong bidi overrides/isolates can reorder displayed text, and the line and
# paragraph separators can forge lines. Weak direction marks (LRM/RLM/ALM) are
# kept: they are legitimate typography for non-Latin text.
_LAYOUT_CONTROLS = frozenset("\u202a\u202b\u202c\u202d\u202e\u2028\u2029\u2066\u2067\u2068\u2069")

_REPLACED_CATEGORIES = frozenset({"Cc", "Cs"})
_REPLACEMENT = "\ufffd"


def single_line(text: str) -> str:
    """Sanitise and collapse whitespace, for logs and error messages."""
    return " ".join(sanitize_text(text).split())


def sanitize_text(text: str) -> str:
    """Replace characters that can corrupt terminals, logs or UTF-8 encoding.

    Keeps printable text, newlines and tabs; normalises CRLF and lone CR to LF.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        character
        if character in "\n\t"
        or (
            unicodedata.category(character) not in _REPLACED_CATEGORIES
            and character not in _LAYOUT_CONTROLS
        )
        else _REPLACEMENT
        for character in normalized
    )
