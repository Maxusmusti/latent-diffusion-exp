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
    sentences = ['Hi', 'This framework generates embeddings for each input sentence', 'Sentences are passed as a list of string.', 'The quick brown fox jumps over the lazy dog.']
    se = SentenceEmbedder()
    embeddings = se.encode_sentences(sentences)

    print(embeddings)
    print(type(embeddings))
    print(embeddings.shape)
