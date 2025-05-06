import re
import spacy

# ---------------------------------------
# Function: detect_script
# Detect the script of a character (Han, Hangul, Latin, etc.)
# Neutral characters (punctuation, digits, whitespace) are treated as "Neutral"
# ---------------------------------------
def detect_script(ch):
    if re.match(r'[\u4e00-\u9fff]', ch):
        return "Han"
    if re.match(r'[\uac00-\ud7af]', ch):
        return "Hangul"
    if re.match(r'[\u0e00-\u0e7f]', ch):
        return "Thai"
    if re.match(r'[\u0600-\u06ff]', ch):
        return "Arabic"
    if re.match(r'[\u0980-\u09ff]', ch):
        return "Bengali"
    if re.match(r'[A-Za-z]', ch):
        return "Latin"
    # Punctuation, digits, and whitespace are treated as neutral
    return "Neutral"

# ---------------------------------------
# Function: split_script_blocks
# Group characters into contiguous chunks based on script identity.
# Neutral characters are merged with the current script block.
# ---------------------------------------
def split_script_blocks(text):
    if not text:
        return []
    blocks = []
    cur_script = detect_script(text[0])
    cur_chunk = text[0]
    for ch in text[1:]:
        scr = detect_script(ch)
        # Neutral characters are not separated; they stay with the current block
        if scr == "Neutral" or scr == cur_script:
            cur_chunk += ch
        else:
            blocks.append((cur_script, cur_chunk))
            cur_chunk, cur_script = ch, scr
    blocks.append((cur_script, cur_chunk))
    return blocks

# ---------------------------------------
# Sentence-splitting rules based on script
# Each rule is a regex pattern defining valid sentence boundaries
# ---------------------------------------
rules = {
    "Han":     r'(?<=[。！？\.\!\?])',
    "Latin":   r'(?<=[\.\!\?])',
    "Hangul":  r'(?<=[\.\!\?])',
    "Thai":    r'(?<=[\.\!\?])',
    "Arabic":  r'(?<=[\.\؟\!\；])',
    "Bengali": r'(?<=[।\?\!])',
    # Neutral: no rule, never split separately
}

# ---------------------------------------
# Function: split_sentences_by_script
# Split each script-based chunk into sub-sentences using language-specific punctuation
# ---------------------------------------
def split_sentences_by_script(blocks):
    out = []
    for script, chunk in blocks:
        pat = rules.get(script)
        if pat:
            parts = [s.strip() for s in re.split(pat, chunk) if s.strip()]
        else:
            # No rule available → treat the entire chunk as one sentence
            parts = [chunk]
        out.extend(parts)
    return out

# ---------------------------------------
# Example test: Mixed-script sentence
# ---------------------------------------
s = "这是내가 가장 좋아하는食物"
blocks = split_script_blocks(s)
print(blocks)
# Output: [('Han', '这是'), ('Hangul', '내가 가장 좋아하는'), ('Han', '食物')]

sents = split_sentences_by_script(blocks)
print(sents)
# Output: ['这是', '내가 가장 좋아하는', '食物']
