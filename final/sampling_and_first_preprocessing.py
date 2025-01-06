#PREPROCESSING AND SAMPLING ON THE ORIGINAL DATASET

import pandas as pd


path = '.\\data\\'
df = pd.read_csv(path + 'en_lyrics.csv')

#dropping useless and redundant columns
df = df.drop(
     columns= [
         'id',
         'language_cld3', 'language_ft', 'language'
     ]
)


df['tag'] = df[
     [
         'is_country', 'is_pop', 'is_rap',
         'is_rb', 'is_rock',
     ]
 ].idxmax(axis=1)


#sampling 
df = df.groupby('tag').sample(frac=0.05, random_state=42)


# remove misc values, write df
df = df.loc[df['tag'] != 'misc']


# explode column tag into is_<genre>
df = pd.concat(
     [
         df,
         pd.get_dummies(df['tag'], prefix='is', dtype=bool)
     ], axis=1
 ).drop(columns= 'tag')


# save sampled dataset -----> PERCHE' E' SALVATO COME DROP_DATASET E NON SAMPLED_DATA (ossia quello su cui abbiamo fatto tutti i procedimenti dopo?)
# df.to_csv(path + 'drop_dataset.csv', index= False)




#----------------------------------------------------------------------------



#COSA SONO QUESTI PASSAGGI SOTTO?

#df = pd.read_csv(path + 'drop_dataset.csv')

# split stanzas, take into account chorus/verse number
def split_into_stanzas(row):
    stanzas = row['lyrics'].split('\n\n')  # Split by double newlines
    stanza_entries = []
    for stanza in stanzas:
        stanza_entry = row.to_dict()  # Convert all metadata to a dictionary
        stanza_entry['stanza'] = stanza.strip()  # Add the stanza
        stanza_entries.append(stanza_entry)
    return stanza_entries

# Transform the dataframe
stanza_data = []
for _, row in df.iterrows():
    stanza_data.extend(split_into_stanzas(row))  # Extend with stanza dictionaries

# Create a new dataframe with stanzas
stanza_df = pd.DataFrame(stanza_data)

# Drop the original 'lyrics' column (optional)
stanza_df = stanza_df.drop(columns=['lyrics'])


# for encoding, sentence_transformers.SentenceTransformer
# seems to be the best fit for the task

# simple explanation: this encodes the stanza as a vector
# of real numbers, which can then be passed to models for
# training and classification

# !IMPORTANT: this is the last step of preprocessing
# for both pipelines