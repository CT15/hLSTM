import re
from urlextract import URLExtract

def replace_url(s):
    extractor = URLExtract()

    if extractor.has_urls(s):
        urls = extractor.find_urls(s, only_unique=True)
        for url in urls:
            s = s.replace(url, "<url>")
        
    return s


def normalize(s):
    # Replace any .!? by a whitespace + the character --> '!' becomes ' !'.
    # \1 means the furst bracketed group --> [,!?]
    # r is to not consider \1 as a character (r to escape a backslash)
    # + means 1 or more
    s = re.sub(r"([.!?])", r" \1", s)
    # Remove any character that is NOT a sequence of lower or upper case letters. + means one or more
    s = re.sub(r"[^a-zA-Z.!?<>]+", r" ", s)
    # Remove a sequence of whitespace characters
    s = re.sub(r"\s+", r" ",s).strip()

    return s


def process_text(s):
    # Assuming that none of the non-html tag contains < or > (Qn: How to be sure?)
    s = re.sub(r'<[^>]*>', '', s) # Replace HTML tags with ""
    s = re.sub('\s+', ' ', s).strip() # Replace multiple spaces with single space
    s = replace_url(s) # replace url with <url>
    s = normalize(s)
    s = s.lower() # change everything to lower case
    
    return s
