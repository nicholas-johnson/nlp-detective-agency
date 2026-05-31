"""Exercise 01 — String Sleuthing (solution)

Slice, search, and normalise raw witness statement strings.
"""


def normalise(text: str) -> str:
    """Strip leading/trailing whitespace and lowercase the entire string."""
    return text.strip().lower()


def word_count(text: str) -> int:
    """Return the number of words (whitespace-separated tokens) in *text*."""
    return len(text.split())


def extract_words(text: str) -> list[str]:
    """Split *text* on whitespace, lowercase each token, and return the list."""
    return [w.lower() for w in text.split()]


def contains_keyword(text: str, keyword: str) -> bool:
    """Return True if *keyword* appears in *text* (case-insensitive)."""
    return keyword.lower() in text.lower()


def censor_word(text: str, word: str) -> str:
    """Replace every occurrence of *word* in *text* with '[CENSORED]' (case-insensitive).

    Performs a simple whole-string replacement — does not handle partial-word
    matches (e.g. censoring 'dock' will also affect 'dockyard').
    """
    lower = text.lower()
    target = word.lower()
    result: list[str] = []
    i = 0
    while i < len(text):
        if lower[i : i + len(target)] == target:
            result.append("[CENSORED]")
            i += len(target)
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def initials(name: str) -> str:
    """Return the uppercased initials of a full name.

    >>> initials("Margaret Hayes")
    'M.H.'
    """
    parts = name.split()
    return ".".join(p[0].upper() for p in parts) + "."


def format_summary(witness: str, case_id: str, word_total: int) -> str:
    """Return a one-line summary string.

    >>> format_summary("Margaret Hayes", "CASE-42", 38)
    'Witness: Margaret Hayes | Case: CASE-42 | Words: 38'
    """
    return f"Witness: {witness} | Case: {case_id} | Words: {word_total}"


def truncate(text: str, max_len: int) -> str:
    """Return *text* truncated to *max_len* characters.

    If truncation occurs, the last three characters are replaced with '...'.
    If *max_len* < 3 return the first *max_len* characters with no ellipsis.
    """
    if len(text) <= max_len:
        return text
    if max_len < 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."
