"""Exercise 02 — Regex Extraction

Extract dates, case IDs, redacted markers, and more from raw case-file text.
Implement each function below — the tests in test_start.py will tell you
when you've got it right.
"""

import re  # noqa: F401 — you'll need this


def find_case_ids(text: str) -> list[str]:
    """Return all case IDs matching the pattern CASE-<digits>.

    >>> find_case_ids("See CASE-42 and CASE-107 for details.")
    ['CASE-42', 'CASE-107']
    """
    # TODO: implement
    pass


def find_dates(text: str) -> list[str]:
    """Return all dates in YYYY-MM-DD format.

    >>> find_dates("Recorded 1947-03-12 and 1948-01-05.")
    ['1947-03-12', '1948-01-05']
    """
    # TODO: implement
    pass


def find_redacted(text: str) -> list[str]:
    """Return all [REDACTED] or [CENSORED] markers (case-insensitive).

    >>> find_redacted("[REDACTED] told me. Also [censored] appeared.")
    ['[REDACTED]', '[censored]']
    """
    # TODO: implement
    pass


def strip_punctuation(text: str) -> str:
    """Remove all characters that are not letters, digits, or whitespace.

    >>> strip_punctuation("Hello, world! Case #42.")
    'Hello world Case 42'
    """
    # TODO: implement
    pass


def extract_statement_ids(text: str) -> list[str]:
    """Return all statement IDs matching STM-<digits>.

    >>> extract_statement_ids("Filed under STM-001 and STM-045.")
    ['STM-001', 'STM-045']
    """
    # TODO: implement
    pass


def mask_names(text: str, names: list[str]) -> str:
    """Replace each name in *names* with '***' (case-insensitive).

    >>> mask_names("Reeves met Hayes at the pub.", ["Reeves", "Hayes"])
    '*** met *** at the pub.'
    """
    # TODO: implement
    pass


def extract_times(text: str) -> list[str]:
    """Return all times in HH:MM (24-hour) or H:MMam/pm format.

    >>> extract_times("At 11:30pm and again at 08:15 the next day.")
    ['11:30pm', '08:15']
    """
    # TODO: implement
    pass


def count_uppercase_words(text: str) -> int:
    """Return the number of fully-uppercase words (2+ letters) in *text*.

    >>> count_uppercase_words("I saw HIM near the DOCKS on Tuesday")
    2
    """
    # TODO: implement
    pass
