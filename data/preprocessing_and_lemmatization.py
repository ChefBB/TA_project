### PREPROCESSING ON THE SAMPLED_DATASET ###

import pandas as pd

path = "./data/sampled_dataset.csv"
df = pd.read_csv(path + "sampled_dataset.csv")


### PRELIMINARY CLEANING PROCESS ON LYRICS ###

#Function to clean the lyrics strings from words inside square brackets that are not keywords for the stanza splitting
import re
keep_pattern = r"\[(Chorus|Verse|Bridge|Intro|Outro|Hook|Prehook|Posthook|Introduction|Interlude|Coda|Conclusion|Refrain).\]*"
def clean_lyrics(lyrics):
    cleaned_lyrics = re.sub(r"\[.*?\]", lambda match: match.group(0) if re.match(keep_pattern, match.group(0)) else "", lyrics)
    return cleaned_lyrics

df['cleaned_lyrics'] = df['lyrics'].apply(clean_lyrics)


#Function to split the stanzas according to the various formats used to denote stanza breaks (keywords with [brackets], without brackets, (keyword), only \n\n between them)    
def process_lyric(ly):
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
    if isinstance(stanza, str):
        pattern = r"^\s*[\[\(]?(Hook|Chorus|Refrain|Bridge)\s*(\]|\))"
        return bool(re.match(pattern, stanza, re.IGNORECASE))

exploded_df["is_chorus"] = (
    exploded_df["stanzas"].apply(is_chorus) | 
    exploded_df.groupby(exploded_df.index)["stanzas"].transform(
        lambda x: x.duplicated(keep=False)
    )
)


#Function that cleans the string header from the keywords in order to get only clean strings (e.g. eliminating "chorus]", "refrain]") and removes \n between single lines
def pulire_stringhe(stanza):
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

#Download the lemmatized_df 
exploded_df.to_csv(path + 'def_lemmatized_df.csv', index= False)






