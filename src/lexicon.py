"""Area B - rule baseline: small hand-written offensive-word lexicon +
edit-distance matcher (catches misspellings such as "idoit", "stupd")."""
from .preprocess import normalize, tokenize, edit_distance

OFFENSIVE_WORDS = [
    "fuck", "fucking", "fucked", "fucker", "motherfucker", "shit", "shitty", "bullshit",
    "bitch", "bitches", "bastard", "asshole", "ass", "dick", "dickhead", "cunt", "piss",
    "pussy", "slut", "whore", "idiot", "idiots", "stupid", "moron", "morons", "dumb",
    "dumbass", "retard", "retarded", "loser", "losers", "scum", "trash", "garbage",
    "hate", "kill", "die", "ugly", "fat", "pathetic", "disgusting", "liar", "liars",
    "crap", "damn", "hell", "suck", "sucks", "jerk", "fool", "fools", "clown", "coward",
    "traitor", "nazi", "nazis", "racist", "racists", "fascist", "creep", "pig", "pigs",
    "stfu", "wtf", "gtfo",
]
LEXICON = set(OFFENSIVE_WORDS)


def match_tokens(text: str):
    """Return the tokens of `text` that match the lexicon (exactly, or within
    edit distance 1 for words of length >= 5)."""
    hits = []
    for tok in tokenize(normalize(text)):
        if tok in LEXICON:
            hits.append(tok)
        elif len(tok) >= 5 and any(abs(len(tok) - len(w)) <= 1 and edit_distance(tok, w) <= 1
                                   for w in LEXICON if len(w) >= 5):
            hits.append(tok)
    return hits


def rule_score(text: str) -> float:
    """'Probability' of offensive = 1 if any lexicon match, else 0."""
    return 1.0 if match_tokens(text) else 0.0
