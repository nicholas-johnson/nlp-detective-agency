"""Exercise 01 — String Sleuthing

Slice, search, and normalise raw witness statement strings.
Implement each function below — the tests in test_start.py will tell you
when you've got it right.
"""


def normalise(text: str) -> str:
    """Strip leading/trailing whitespace and lowercase the entire string."""
    # TODO: implement
    pass


def word_count(text: str) -> int:
    """Return the number of words (whitespace-separated tokens) in *text*."""
    # TODO: implement
    pass


def extract_words(text: str) -> list[str]:
    """Split *text* on whitespace, lowercase each token, and return the list."""
    # TODO: implement
    pass


def contains_keyword(text: str, keyword: str) -> bool:
    """Return True if *keyword* appears in *text* (case-insensitive)."""
    # TODO: implement
    pass


def censor_word(text: str, word: str) -> str:
    """Replace every occurrence of *word* in *text* with '[CENSORED]' (case-insensitive).

    Performs a simple whole-string replacement — does not handle partial-word
    matches (e.g. censoring 'dock' will also affect 'dockyard').
    """
    # TODO: implement
    pass


def initials(name: str) -> str:
    """Return the uppercased initials of a full name.

    >>> initials("Margaret Hayes")
    'M.H.'
    """
    # TODO: implement
    pass


def format_summary(witness: str, case_id: str, word_total: int) -> str:
    """Return a one-line summary string.

    >>> format_summary("Margaret Hayes", "CASE-42", 38)
    'Witness: Margaret Hayes | Case: CASE-42 | Words: 38'
    """
    # TODO: implement
    pass


def truncate(text: str, max_len: int) -> str:
    """Return *text* truncated to *max_len* characters.

    If truncation occurs, the last three characters are replaced with '...'.
    If *max_len* < 3 return the first *max_len* characters with no ellipsis.
    """
    # TODO: implement
    pass
