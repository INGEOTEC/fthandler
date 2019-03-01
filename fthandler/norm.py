import re
import os
import sys
import json
import gzip
import unicodedata

# "'" is not here to add a simple support for contractions
# '@' is left to recognize users, the same for '_'
# the asterisk is also reserved to allow some masked words like 'f***'

NOSTART = "*•\"'“”…’‘–+→¨"
NOEND = "•\"'“”…’‘–+→¨"
PUNCTUACTION = ";:,.\\-\"/$…" + NOSTART + NOEND 
SYMBOLS = "()[]¿?¡!{}~<>|--^~«»<>"
SET_SYMBOLS = set(PUNCTUACTION + SYMBOLS)
EMO = None

with open(os.path.join(os.path.dirname(__file__), "emojis.txt")) as f:
    EMOTICONS = set([line.strip() for line in f.readlines()])


def normalize_words(text, mask_users=True, mask_hashtags=True, mask_nums=True):
    arr = text.split()
    L = []
    for t in arr:
        if mask_nums and t.isdigit():
            L.append('_num')
        elif mask_users and t[0] == '@':
            L.append('_usr')
        elif mask_hashtags and t[0] == '#':
            L.append('_htag')
        else:
            if t[0] in NOSTART:
                t = t[1:]
            if len(t) == 0:
                continue
            if t[-1] in NOEND:
                t = t[:-1]
            if len(t) == 0:
                continue

            L.append(t)

    return " ".join(L)
    
    
def normalize(text, del_diac=True, del_punc=False, emoticons_to_emojis=True, mask_urls=True, mask_users=True, mask_hashtags=True, mask_nums=True, mask_rep=True):
    L = []

    text = text.lower().replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('\n', ' ')

    if emoticons_to_emojis:
        text = re.sub(r'[:;8=][\`\-\'\"\^]?[\)dp]', chr(0x1f604), text)
        text = re.sub(r'[:;8=][\`\-\'\"\^]?[\(c]', chr(0x1f61f), text)
        text = re.sub(r':-?\||:\^\|', chr(0x1f612), text)
    
    if mask_urls:
        text = re.sub(r'https?:/\S+', '_url', text)

    BLANK = ' '
    prev = BLANK
    count_prev = 1

    for u in unicodedata.normalize('NFD', text):
        if del_diac:
            o = ord(u)
            if 0x300 <= o and o <= 0x036F:
                if o == 0x303:  # tilde
                    L.append(u)
                    prev = u
                    continue
                else:
                    continue

        if u in EMOTICONS:
            if prev != BLANK:
                L.append(BLANK)

            L.append(u)
            L.append(BLANK)
            prev = BLANK
            continue

        elif u in ('\r', ' ', '\t'):
            u = BLANK
            if prev == u:
                continue

        elif u in SET_SYMBOLS:
            if del_punc:
                # prev = u
                continue
            else:
                if prev != BLANK:
                    L.append(BLANK)

                L.append(u)
                L.append(BLANK)
                prev = BLANK
                continue
        
        if mask_rep:
            if prev == u:
                count_prev += 1
                if count_prev == 3:
                    L[-1] = '*'
                    continue
                elif count_prev > 3:
                    continue
            else:
                count_prev = 1

        prev = u
        L.append(u)

    return normalize_words("".join(L), mask_users=mask_users, mask_hashtags=mask_hashtags, mask_nums=mask_nums)


def encode_prediction(lines):
    out = []
    for line in lines:
        arr = line.replace('__label__', '').strip().split()
        L = []
        while len(arr):
            v = arr.pop()
            k = arr.pop()
            L.append((k, v))

        L.sort(key=lambda x: x[0])
        L = [float(x[1]) for x in L]
        # arr = line.rstrip().split('__label__')[1:]
        # print(" ".join(L))
        if len(L) > 0:
            out.append(L)

    return out
