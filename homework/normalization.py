"""Canonical text form shared by matching and rubric validation."""
import unicodedata


def normalize(value):
    value = unicodedata.normalize('NFKC', value).casefold()
    return ''.join(c for c in value if c.isalnum())
