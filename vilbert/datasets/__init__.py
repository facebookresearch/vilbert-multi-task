from .concept_cap_dataset import ConceptCapLoaderTrain, ConceptCapLoaderVal, ConceptCapLoaderRetrieval
from .foil_dataset import FoilClassificationDataset
from .vqa_dataset import VQAClassificationDataset
from .vqa_mc_dataset import VQAMultipleChoiceDataset
from .refer_expression_dataset import ReferExpressionDataset
from .retreival_dataset import RetreivalDataset, RetreivalDatasetVal
from .vcr_dataset import VCRDataset
from .visdial_dataset import VisDialDataset
from .refer_dense_caption import ReferDenseCpationDataset

# from .flickr_retreival_dataset import FlickrRetreivalDatasetTrain, FlickrRetreivalDatasetVal

__all__ = ["FoilClassificationDataset", \
		   "VQAClassificationDataset", \
		   "VQAMultipleChoiceDataset", \
		   "ConceptCapLoaderTrain", \
		   "ConceptCapLoaderVal", \
		   "ReferExpressionDataset", \
		   "RetreivalDataset", \
		   "RetreivalDatasetVal",\
		   "VCRDataset", \
		   "VisDialDataset", \
		   "ConceptCapLoaderRetrieval", \
		   "ReferDenseCpationDataset"]

DatasetMapTrain = {
				   'VQA': VQAClassificationDataset,
				   'VisualDialog': VisDialDataset,
				   'VCR_Q-A': VCRDataset,
				   'VCR_QA-R': VCRDataset,				   
				   'RetrievalCOCO': RetreivalDataset,
				   'RetrievalFlickr30k': RetreivalDataset,
				   'refcoco': ReferExpressionDataset,
				   'refcoco+': ReferExpressionDataset,
				   'refgoogle': ReferDenseCpationDataset	   
				   }		


DatasetMapEval = {
				 'VQA': VQAClassificationDataset,
				 'VisualDialog': VisDialDataset,
				 'VCR_Q-A': VCRDataset,
				 'VCR_QA-R': VCRDataset,				   
				 'RetrievalCOCO': RetreivalDatasetVal,
				 'RetrievalFlickr30k': RetreivalDatasetVal,
				 'refcoco': ReferExpressionDataset,			   
				 'refcoco+': ReferExpressionDataset,
				 'refgoogle': ReferDenseCpationDataset
				}