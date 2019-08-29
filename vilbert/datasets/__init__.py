from .concept_cap_dataset import (
    ConceptCapLoaderTrain,
    ConceptCapLoaderVal,
    ConceptCapLoaderRetrieval,
)
from .foil_dataset import FoilClassificationDataset
from .vqa_dataset import VQAClassificationDataset
from .vqa_mc_dataset import VQAMultipleChoiceDataset
from .refer_expression_dataset import ReferExpressionDataset
from .retreival_dataset import RetreivalDataset, RetreivalDatasetVal
from .vcr_dataset import VCRDataset
from .visdial_dataset import VisDialDataset
from .visual_entailment_dataset import VisualEntailmentDataset
from .refer_dense_caption import ReferDenseCpationDataset
from .visual_genome_dataset import GenomeQAClassificationDataset

# from .flickr_retreival_dataset import FlickrRetreivalDatasetTrain, FlickrRetreivalDatasetVal
__all__ = [
    "FoilClassificationDataset",
    "VQAClassificationDataset",
    "GenomeQAClassificationDataset",
    "VQAMultipleChoiceDataset",
    "ConceptCapLoaderTrain",
    "ConceptCapLoaderVal",
    "ReferExpressionDataset",
    "RetreivalDataset",
    "RetreivalDatasetVal",
    "VCRDataset",
    "VisDialDataset",
    "VisualEntailmentDataset",
    "ConceptCapLoaderRetrieval",
]

DatasetMapTrain = {
    "VQA": VQAClassificationDataset,
    "GenomeQA": GenomeQAClassificationDataset,
    "VisualDialog": VisDialDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalCOCO": RetreivalDataset,
    "RetrievalFlickr30k": RetreivalDataset,
    "refcoco": ReferExpressionDataset,
    "refcoco+": ReferExpressionDataset,
    "refcocog": ReferExpressionDataset,
    "VisualEntailment": VisualEntailmentDataset,
}


DatasetMapEval = {
    "VQA": VQAClassificationDataset,
    "GenomeQA": GenomeQAClassificationDataset,
    "VisualDialog": VisDialDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalCOCO": RetreivalDatasetVal,
    "RetrievalFlickr30k": RetreivalDatasetVal,
    "refcoco": ReferExpressionDataset,
    "refcoco+": ReferExpressionDataset,
    "refcocog": ReferExpressionDataset,
    "VisualEntailment": VisualEntailmentDataset,
}
