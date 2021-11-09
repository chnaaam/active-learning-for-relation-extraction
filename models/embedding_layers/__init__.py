from dataclasses import dataclass

from .bert import BertEmbeddingLayer
from .distilbert import DistilBertEmbeddingLayer
from .electra import ElectraEmbeddingLayer
from .embedding_layer import EmbeddingLayer

@dataclass
class EmbeddingLayerFactories:
    BERT = BertEmbeddingLayer
    DISTILBERT = DistilBertEmbeddingLayer
    ELECTRA = ElectraEmbeddingLayer

    get_layer = {
        "BERT": BERT,
        "DISTILBERT": DISTILBERT,
        "ELECTRA": ELECTRA
    }


__all__ = [
    "EmbeddingLayerFactories", 
    "EmbeddingLayer"
]