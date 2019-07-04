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

DatasetMapTrain = {'TASK0': ConceptCapLoaderTrain,
				   'TASK1': VQAClassificationDataset,
				   'TASK2': VQAMultipleChoiceDataset,
				   'TASK4': VisDialDataset,
				   'TASK6': VCRDataset,
				   'TASK7': VCRDataset,				   
				   'TASK8': RetreivalDataset,
				   'TASK9': RetreivalDataset,
				   'TASK11': ReferExpressionDataset,
				   'TASK12': ReferExpressionDataset,
				   'TASK13': ReferDenseCpationDataset	   
				   }		

DatasetMapVal = {'TASK0': ConceptCapLoaderVal,
				 'TASK1': VQAClassificationDataset,
				 'TASK2': VQAMultipleChoiceDataset,
				 'TASK4': VisDialDataset,
				 'TASK6': VCRDataset,
				 'TASK7': VCRDataset,				   
				 'TASK8': RetreivalDataset,
				 'TASK9': RetreivalDataset,
				 'TASK11': ReferExpressionDataset,	
				 'TASK12': ReferExpressionDataset,
				 'TASK13': ReferDenseCpationDataset	   
				}

DatasetMapTest = {'TASK0': ConceptCapLoaderVal,
				 'TASK1': VQAClassificationDataset,
				 'TASK2': VQAMultipleChoiceDataset,
				 'TASK4': VisDialDataset,
				 'TASK6': VCRDataset,
				 'TASK7': VCRDataset,				   
				 'TASK8': RetreivalDatasetVal,
				 'TASK9': RetreivalDatasetVal,
				 'TASK11': ReferExpressionDataset,			   
				 'TASK12': ReferExpressionDataset,
				 'TASK13': ReferDenseCpationDataset
				}