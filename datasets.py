import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import Entrez, Medline


def clean_text_pubmed(text):
    text = re.sub(r'\s+', ' ', text)     # Remove excessive whitespace
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)  # Keep alpha-numeric and basic punctuations
    return text.strip()

def clean_text_medquad(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_symcat_df(df, has_consult_id=False):
    # If 'consult_id' exists, ensure it's of string type
    if has_consult_id and 'consult_id' in df.columns:
        df['consult_id'] = df['consult_id'].astype(str)
    # Clean 'disease_tag'
    # Convert 'disease_tag' to string type first, handling potential non-string values
    df['disease_tag'] = df['disease_tag'].astype(str).str.lower().str.strip()
    # Ensure 'explicit_symptoms' and 'implicit_symptoms' are lists of lowercase strings
    for col in ['explicit_symptoms', 'implicit_symptoms']:
        df[col] = df[col].apply(lambda x: [s.lower().strip() for s in x] if isinstance(x, list) else [])
    # print(len(df))
    return df

def pubmed_dataset():
    Entrez.email = "arunpwr2712@gmail.com"  # Set your email
    search_term = "common cold AND humans[MeSH Terms]"  # Example search: common cold
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=100)  # Search PubMed and fetch IDs
    record = Entrez.read(handle)
    handle.close()
    id_list = record["IdList"]

    # Fetch summaries
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    pubmed_data = list(records)
    handle.close()

    # Convert to DataFrame
    df_pubmed = pd.DataFrame(pubmed_data)
    df_pubmed = df_pubmed[['TI', 'AB']]  # Keep only title and abstract
    df_pubmed.dropna(inplace=True)       # Remove empty abstracts
    df_pubmed.rename(columns={'TI': 'Title', 'AB': 'Abstract'}, inplace=True)
    df_pubmed['Title'] = df_pubmed['Title'].apply(clean_text_pubmed)
    df_pubmed['Abstract'] = df_pubmed['Abstract'].apply(clean_text_medquad)

    return df_pubmed


# Load the medquad dataset
df_medquad = pd.read_csv("C:/Users/arunp/Documents/project/M.Tech Mini Project/MedGPT/datasets/medquad.csv")
df_medquad.dropna(subset=['question', 'answer'], inplace=True)  # Drop rows with missing 'Question' or 'Answer'
df_medquad['question'] = df_medquad['question'].apply(clean_text_medquad)  # Apply cleaning to 'Question' and 'Answer' columns
df_medquad['answer'] = df_medquad['answer'].apply(clean_text_medquad)  # Apply cleaning to 'Question' and 'Answer' columns


# Load the symcat datasets
df_train_symcat = pd.read_pickle("C:/Users/arunp/Documents/project/M.Tech Mini Project/MedGPT/datasets/symcat_400_train_df.pkl")
df_train = preprocess_symcat_df(df_train_symcat)
#Load the pubmed dataset
df_pubmed=pubmed_dataset()










# Display basic information
# print(df_medquad.head())
# print(df_medquad.columns)
# Check for missing values
# print(df_medquad.isnull().sum())
# df_medquad.dropna(subset=['question', 'answer'], inplace=True)

# # Apply cleaning to 'Question' and 'Answer' columns
# df_medquad['question'] = df_medquad['question'].apply(clean_text_medquad)
# df_medquad['answer'] = df_medquad['answer'].apply(clean_text_medquad)


# # Load the datasets
# df_train_symcat = pd.read_pickle("C:/Users/arunp/Documents/project/M.Tech Mini Project/MedGPT/datasets/symcat_400_train_df.pkl")

# # Display the number of records in each set
# print(f"Training set: {df_train_symcat.shape}")
# print(df_train_symcat.columns)

# # Plotting symcat
# # Count of each disease in the training set
# disease_counts = df_train_symcat['disease_tag'].value_counts().head(10)

# plt.figure(figsize=(10, 6))
# sns.barplot(x=disease_counts.values, y=disease_counts.index, palette='viridis')
# plt.title('Top 10 Diseases in Training Set')
# plt.xlabel('Frequency')
# plt.ylabel('Disease')
# plt.tight_layout()
# plt.show()



# # plotting pubmed
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# text = ' '.join(pubmed_df['Abstract'].tolist())
# wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(text)

# plt.figure(figsize=(15, 7))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title("Common Terms in PubMed Abstracts")
# plt.show()

# from collections import Counter
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# stop_words = set(stopwords.words('english'))
# words = word_tokenize(text.lower())
# filtered_words = [w for w in words if w.isalpha() and w not in stop_words]
# word_freq = Counter(filtered_words)

# # Top 20 terms
# top_terms = word_freq.most_common(20)
# terms, freqs = zip(*top_terms)

# plt.figure(figsize=(12, 6))
# plt.bar(terms, freqs)
# plt.xticks(rotation=45)
# plt.title("Top 20 Words in PubMed Abstracts")
# plt.xlabel("Terms")
# plt.ylabel("Frequency")
# plt.show()