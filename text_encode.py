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
    # From https://www.sbert.net/

    sentences = ['White dog running']
    se = SentenceEmbedder()
    embeddings = se.encode_sentences(sentences)

    print(embeddings)
    print(type(embeddings))
    print(embeddings.shape)

run_example()
