
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification



tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
config = AutoConfig.from_pretrained('dmis-lab/biobert-v1.1', num_labels=3)
model = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1',config = config)


model.save_pretrained("/workspace/amit_pg/Bio-BERT/model")
tokenizer.save_pretrained('/workspace/amit_pg/Bio-BERT/tokenizer')
config.save_pretrained('/workspace/amit_pg/Bio-BERT/config')

