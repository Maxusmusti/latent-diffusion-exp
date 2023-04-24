from sentence_transformers import SentenceTransformer

class SentenceEmbedder():
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode_sentences(self, sentences: list):
        """
        Generate size 384 embedding per sentence
        """
        embeddings = self.model.encode(sentences)
        return embeddings

def run_example():
    # From - https://www.sbert.net/
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #Our sentences we like to encode
    sentences = ['This framework generates embeddings for each input sentence',
        'Sentences are passed as a list of string.',
        'The quick brown fox jumps over the lazy dog.']

    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    #Print the embeddings
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")
