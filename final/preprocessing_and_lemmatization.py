### PREPROCESSING ON THE SAMPLED_DATASET ###

import pandas as pd

path = "./models/data/"
df = pd.read_csv(path + "sampled_dataset.csv")


### PRELIMINARY CLEANING PROCESS ON LYRICS ###

#Function to clean the lyrics strings from words inside square brackets that are not keywords for the stanza splitting
import re
keep_pattern = r"\[(Chorus|Verse|Bridge|Intro|Outro|Hook|Prehook|Posthook|Introduction|Interlude|Coda|Conclusion|Refrain).\]*"

def clean_lyrics(lyrics):
    """
    Cleans song lyrics by removing specific bracketed text based on a pattern.

    This function uses regular expressions to identify text within square brackets (e.g., "[Chorus]"). 
    It removes the text unless it matches a predefined pattern (`keep_pattern`), in which case it retains the text.

    Args:
        lyrics (str): A string containing the song lyrics to be cleaned.

    Returns:
        str: The cleaned lyrics with unwanted bracketed text removed.
        """
    cleaned_lyrics = re.sub(r"\[.*?\]", lambda match: match.group(0) if re.match(keep_pattern, match.group(0)) else "", lyrics)
    return cleaned_lyrics

df['cleaned_lyrics'] = df['lyrics'].apply(clean_lyrics)


#Function to split the stanzas according to the various formats used to denote stanza breaks (keywords with [brackets], without brackets, (keyword), only \n\n between them)    
def process_lyric(ly):
    """
    Processes a song lyric by splitting it into sections based on various patterns and cleaning up whitespace.
    The function identifies and splits by specific stanza labels such as Verse, Chorus, Intro, etc. and further
    splits the lyrics to isolate individual stanzas. Empty sections are removed.

    Args:
        ly (str): A string containing the full lyrics of a song, including section labels and the lyrics themselves.

    Returns:
        list of str: A list of processed lyrics sections (stanzas) after splitting by various patterns and removing
                     extra whitespace. Empty sections are excluded.

    The function performs the following operations:
    - Identifies and splits the lyrics based on section labels (e.g., Verse, Chorus, etc.).
    - Further splits the sections based on specific text patterns like "Verse" or newline characters.
    - Cleans up whitespace by stripping each section.
    - Removes any empty sections resulting from the splits.
    """
    sections = [ly.strip()]
    
    #Split by the pattern
    pattern = r"\[(Introduction|Interlude|Coda|Hook|Prehook|Posthook|Conclusion|Refrain|Verse|Intro|Outro|Chorus|Bridge).\]*"
    if re.search(pattern, ly, re.IGNORECASE):
        sections = [segment for section in sections for segment in section.split("\n[")]

    #Further split by "Verse"
    if re.search(r"Verse", ly, re.IGNORECASE):
        sections = [segment for section in sections for segment in section.split("Verse")]

    #Further split by \n\n
    if "\n\n" in ly:
        sections = [segment for section in sections for segment in section.split("\n\n")]
    
    #Further split by "(Chorus)"
    if re.search(r"\(Chorus\)", ly, re.IGNORECASE):
        sections = [segment for section in sections for segment in section.split("(")]

    #Clean whitespace for each stanza and remove empty ones
    return [section.strip() for section in sections if section.strip()]


df["processed_lyrics"] = df["cleaned_lyrics"].apply(process_lyric)


#Function to delete strings there are uninformative (only the keyword, empty strings, strings that are too short)
def clean_list(lyrics_list):
    """
    Removes uninformative strings from a list of lyrics.
    The function deletes lines that are either:
    - Only contain a section label (e.g., "Hook", "Chorus", "Verse", etc.).
    - Empty strings.
    - Strings that are too short (less than or equal to 20 characters).

    Args:
        lyrics_list (list): A list of strings representing the sections of a song.

    Returns:
        list: A cleaned list of strings where uninformative lines have been removed. 
    """
    if not isinstance(lyrics_list, list):
        return lyrics_list
    tag_pattern = r"^\s*(Hook|Chorus|Bridge|Verse|Outro|Intro|Refrain|Prehook|Posthook|Coda|Interlude|Conclusion).*?\]\s*$|^(.{,20})\s*$"
    
    return [line for line in lyrics_list if line and not re.match(tag_pattern, line, re.IGNORECASE)]

df['cleaned_lists'] = df["processed_lyrics"].apply(clean_list)


### EXPLODE SAMPLED_DATA ###

#Removing redundant variables
df = df.drop(['cleaned_lyrics', 'processed_lyrics'], axis=1)

#Create n records for the n stanzas of each song
exploded_df = df.explode('cleaned_lists', ignore_index=False)
exploded_df.rename(columns={'cleaned_lists': 'stanzas'}, inplace=True)

#Numbers the stanzas based on their order
exploded_df['stanza_number'] = exploded_df.groupby(exploded_df.index).cumcount()



### DEEPER CLEANING PROCESS ON LYRICS ###

#Function that creates a new variable "is_chorus" with a binary value T/F with true if it's a stanza with a specific header and/or whether a stanza is repeated for the same song (index)
def is_chorus(stanza):
    """
    Determines whether a given stanza is a chorus (or similar section) based on its content.

    Args:
        stanza (str): A string representing a stanza or section of song lyrics.

    Returns:
        bool: True if the stanza contains a "Chorus", "Hook", "Refrain", or "Bridge" section label at the beginning,
              False otherwise.
    """
    if isinstance(stanza, str):
        pattern = r"^\s*[\[\(]?(Hook|Chorus|Refrain|Bridge)\s*(\]|\))"
        return bool(re.match(pattern, stanza, re.IGNORECASE))

# Add a new column "is_chorus" to flag whether each stanza is part of a chorus:
# 1. Apply the `is_chorus` function to check if a stanza explicitly contains section labels like "[Chorus]" or "(Hook)".
# 2. Group stanzas by their original song (using the DataFrame index) and flag repeated stanzas 
#    (using `.duplicated(keep=False)`) since choruses are often repeated within a song.
# 3. Combine these two conditions with a logical OR (`|`), marking a stanza as a chorus if it matches 
#    either explicit labels or repetition within the song.
exploded_df["is_chorus"] = (
    exploded_df["stanzas"].apply(is_chorus) | 
    exploded_df.groupby(exploded_df.index)["stanzas"].transform(
        lambda x: x.duplicated(keep=False)
    )
)


#Function that cleans the string header from the keywords in order to get only clean strings (e.g. eliminating "chorus]", "refrain]") and removes \n between single lines
def pulire_stringhe(stanza):
    """
    Cleans the input string (stanza) by removing keywords like "chorus]", "refrain]" or any similar section labels,
    and replaces newlines between single lines with a space.

    Args:
        stanza (str): A string representing a stanza or section of song lyrics to be cleaned.

    Returns:
        str: A cleaned string where the section labels (e.g., "chorus]", "refrain]") are removed,
             and newlines between single lines are replaced by a space.
    """
    pattern = r"^.*?\]|\)\n*"
    
    result1 = re.sub(pattern, "", stanza)
    result2 = re.sub(r"\n", " ", result1)
    
    return result2


exploded_df["stanzas"] = exploded_df["stanzas"].apply(pulire_stringhe)



### DROPPING DUPLICATE RECORDS ###
exploded_df = exploded_df.drop_duplicates(subset=["title", "artist", "stanzas", "is_chorus"])



### LEMMATIZATION ###
import spacy

#Some packages have been disabled to reduce the running time since they are not useful for this task
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])

def tokenize(stanza):
    """
    Tokenizes a given stanza by converting it to lowercase, processing it with a spaCy NLP model,
    and extracting its lemmatized tokens, excluding punctuation and spaces.

    Args:
        stanza (str): A string representing a stanza or section of song lyrics to be tokenized.

    Returns:
        list of str: A list of lemmatized tokens (words) from the input stanza, excluding punctuation
                     and spaces.
    """
    if isinstance(stanza, str):
        stanza = stanza.lower()
        doc = nlp(stanza)
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space ]
        return tokens

exploded_df["lemmatized_stanzas"] = exploded_df["stanzas"].apply(tokenize)


### CREATION OF THE LEMMATIZED_DF ###

#dropping redundant variables
exploded_df = exploded_df.drop(["lyrics", "stanzas"], axis=1)

#Download the lemmatized_df 
#exploded_df.to_csv(path + 'def_lemmatized_df.csv')

#Rename a variable
#exploded_df1 = pd.read_csv(path + 'def_lemmatized_df.csv')

#Rename a variable to indicate the id
exploded_df.rename(
    columns= {'Unnamed: 0' : 'id'},
    inplace= True
)

#Download the lemmatized_df; this step produces a lemmatized df 
#exploded_df.to_csv(path + 'def_lemmatized_df.csv', index= False)






