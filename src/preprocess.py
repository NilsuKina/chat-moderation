"""Area B - Classical NLP: regex normalization, tokenization, edit distance.

normalize()  : cleans noisy chat text (URLs, @mentions, leetspeak, repeated letters,
               Turkish characters) with regular expressions.
tokenize()   : simple regex word tokenizer.
edit_distance(): minimum edit distance (Levenshtein, dynamic programming).
"""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")          # "fuuuuck" -> "fuuck"
NON_WORD_RE = re.compile(r"[^a-z0-9\s'*]")
SPACES_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9*']+")

# leetspeak characters that appear *inside* words: "sh1t", "@ss", "b!tch", "a$$"
LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
            "@": "a", "$": "s", "!": "i"}
WORD_CHUNK_RE = re.compile(r"\S+")

# Turkish characters -> ASCII (lets the same pipeline work on Turkish chat)
TR_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def _deleet(match):
    word = match.group(0)
    if not re.search(r"[a-z]", word):        # pure numbers / symbols stay as they are
        return word
    core = word.rstrip("!?.,")               # "idiot!!!" -> keep "!!!" as punctuation
    return "".join(LEET_MAP.get(c, c) for c in core) + word[len(core):]


def normalize(text: str) -> str:
    text = text.translate(TR_MAP).lower()
    text = URL_RE.sub(" url ", text)
    text = USER_RE.sub(" user ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = WORD_CHUNK_RE.sub(_deleet, text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = NON_WORD_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()


def tokenize(text: str):
    return TOKEN_RE.findall(text)


def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance with insert/delete/substitute cost 1 (DP table)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,              # deletion
                           cur[j - 1] + 1,           # insertion
                           prev[j - 1] + (ca != cb)))  # substitution
        prev = cur
    return prev[-1]


if __name__ == "__main__":
    for s in ["@USER you are such an 1d10t!!! http://x.co",
              "shuuuut up you b!tch",
              "Çok güzel bir oyun #gg"]:
        print(repr(s), "->", repr(normalize(s)), tokenize(normalize(s)))
    print(edit_distance("idoit", "idiot"))
