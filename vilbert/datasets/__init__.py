from .concept_cap_dataset import ConceptCapLoaderTrain, ConceptCapLoaderVal, ConceptCapLoaderRetrieval
from .foil_dataset import FoilClassificationDataset
from .vqa_dataset import VQAClassificationDataset
from .refer_expression_dataset import ReferExpressionDataset
from .retreival_dataset import RetreivalDatasetTrain, RetreivalDatasetVal
from .vcr_dataset import VCRDataset
from .visdial_dataset import VisDialDataset
# from .flickr_retreival_dataset import FlickrRetreivalDatasetTrain, FlickrRetreivalDatasetVal

__all__ = ["FoilClassificationDataset", \
		   "VQAClassificationDataset", \
		   "ConceptCapLoaderTrain", \
		   "ConceptCapLoaderVal", \
		   "ReferExpressionDataset", \
		   "RetreivalDatasetTrain", \
		   "RetreivalDatasetVal",\
		   "VCRDataset", \
		   "VisDialDataset", \
		   "ConceptCapLoaderRetrieval"]

DatasetMapTrain = {'TASK0': ConceptCapLoaderTrain,
				   'TASK1': VQAClassificationDataset,
				   }		

DatasetMapVal = {'TASK0': ConceptCapLoaderVal,
				 'TASK1': VQAClassificationDataset,
				}