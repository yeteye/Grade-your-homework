"""Canonical text form shared by matching and rubric validation."""
import unicodedata


def normalize(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())
