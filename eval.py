import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from torch import cuda
from seqeval.metrics import classification_report
import torch.nn.functional as F
import os
import torch.nn as nn

MAX_LEN = 256
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
lab_ids = []
tokenizer_dir = "/workspace/amit_pg/Bio-BERT/tokenizer"
model_dir = "/workspace/amit_pg/Bio-BERT/model"
config_dir = "/workspace/amit_pg/Bio-BERT/config"
num_labels = 0
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def loadData():
    test_data = pd.read_csv('BC5CDR-disease/test.tsv',sep='\t',quoting=csv.QUOTE_NONE,names=["Tokens","Labels"],skip_blank_lines = False)
    train_data = pd.read_csv('BC5CDR-disease/train.tsv',sep='\t',quoting=csv.QUOTE_NONE,names=["Tokens","Labels"],skip_blank_lines = False)
    return test_data,train_data

def IdToLabelAndLabeltoId(train_data):
    label_list = train_data["Labels"]
    label_list = [*set(label_list)]
    num_labels = len(label_list)
    label_list = [x for x in label_list if not pd.isna(x)]
    id2label = {0 : 'B', 1 : 'I' , 2 : 'O' }
    label2id = { id2label[id]: id for id in id2label}
    return id2label,label2id





def convert_to_sentence(df):
    sent = ""
    sent_list = []
    label = ""
    label_list = []
    for tok,lab in df.itertuples(index = False):
        if pd.isna(tok) and pd.isna(lab):
            sent = sent[1:]
            sent_list.append(sent)
            sent = ""
            label = label[1:]
            label_list.append(label)
            label = ""
        else:
            sent = sent + " " +str(tok)
            label = label+ "," + str(lab)
    if sent != "":
        sent_list.append(sent)
        label_list.append(label)
    return sent_list,label_list




def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []
    sentence = str(sentence)
    sentence = sentence.strip()
    text_labels = str(text_labels)

    for word, label in zip(sentence.split(), text_labels.split(",")):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label])
        labels.extend(['X'] * (n_subwords-1))
    #print(tokenized_sentence)
    #print(labels)
    return tokenized_sentence, labels

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len,label2id,id2label):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.id2label = id2label
        
    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.data.Sentence[index]  
        word_labels = self.data.Labels[index]
        label2id = self.label2id
       
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
        
        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        labels.insert(0, "X") # add outside label for [CLS] token
        labels.insert(-1, "X") # add outside label for [SEP] token

        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
          # truncate
          tokenized_sentence = tokenized_sentence[:maxlen]
          labels = labels[:maxlen]
        else:
          # pad
          tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
          labels = labels + ["X" for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        
        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        #print(labels)
        label_ids = [label2id[label] if label != "X" and label != '' else -100 for label in labels]
        # the following line is deprecated
        #label_ids = [label if label != 0 else -100 for label in label_ids]
        lab_ids.append(label_ids)
        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(attn_mask, dtype=torch.long),
              #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
              'targets': torch.tensor(label_ids, dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len


def Test(model, testing_loader,device,id2label):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    e_preds,e_labels = [],[]
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)
            
            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs[0], outputs[1]
            eval_logits = F.softmax(eval_logits, dim=2)
            eval_loss += loss.item()

            nb_eval_steps += 1
            
            nb_eval_examples += targets.size(0)
            d = eval_logits
            eval_preds.extend(d.cpu().numpy())
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Testing loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            
            
            e_labels.extend(targets)
            e_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy
    
    #print(e_labels[-1])
    #print(e_preds[-1])

    #labels = [[id2label[id.item()]  for id in l ]for l in eval_labels]
    #predictions = [[id2label[id.item()] for id in l] for l in eval_preds]
    
    l = [id2label[id.item()]if id.item() != -100 else "O" for id in e_labels]
    p = [id2label[id.item()]if id.item() != -100 else "O" for id in e_preds]

    #print(len(e_labels))
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Testing Loss: {eval_loss}")
    print(f"Testing Accuracy: {eval_accuracy}")
    
    return eval_preds,l,p
    
def align_predictions(predictions, label_ids,id2label):
     preds = np.argmax(predictions, axis=2)
     batch_size, seq_len = preds.shape
     label_ids = np.array(label_ids)
     out_label_list = [[] for _ in range(batch_size)]
     preds_list = [[] for _ in range(batch_size)]
        
     for i in range(batch_size):
         for j in range(seq_len):
             if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                 out_label_list[i].append(id2label[label_ids[i][j]])
                 preds_list[i].append(id2label[preds[i][j]])

     return preds_list, out_label_list
    
def categorize(preds_list):
    ind = 0
    output_test_predictions_file = "test_predictions.txt"
    if True:
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join('./BC5CDR-disease/', "test.tsv"), "r") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not preds_list[example_id]:
                            example_id += 1
                    elif preds_list[example_id]:
                        entity_label = preds_list[example_id].pop(0)
                        if entity_label == 'O':
                            output_line = line.split()[0] + " " + entity_label + "\n"
                        else:
                            output_line = line.split()[0] + " " + entity_label[0] + "\n"
                       # output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                        writer.write(output_line)
                     
               
def main():

    #inp = ["Famotidine is a histamine H2 - receptor antagonist used in inpatient settings for prevention of stress ulcers and is showing increasing popularity because of its low cost."]
    #l = ["O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,B,O,O,O,O,O,O,O,O,O,O,O"]
    test_data,train_data = loadData()
    
    id2label,label2id = IdToLabelAndLabeltoId(train_data)
    
    test_sent,test_label = convert_to_sentence(test_data)
    #devel_sent,devel_label = convert_to_sentence(devel_data)
    
    print("Data loaded")

    #tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    config = AutoConfig.from_pretrained(
        config_dir,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir
    )

    test_data = {'Sentence' : test_sent,'Labels' : test_label}
    test_data = pd.DataFrame(test_data)
    

    #train_data = train_data[["Sentence", "Labels"]].drop_duplicates().reset_index(drop=True)
    #test_data = test_data[["Sentence", "Labels"]].drop_duplicates().reset_index(drop=True)
   
    print("Data preprocessed")

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': TEST_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0
                    }
    testing_set = dataset(test_data, tokenizer, MAX_LEN,label2id,id2label)
    testing_loader = DataLoader(testing_set, **test_params)
    print("Ready for model")
    
    device = 'cuda:0' if cuda.is_available() else 'cpu'
    
    
    '''
    #model = AutoModelForTokenClassification.from_pretrained(model_dir, 
                                                    num_labels=len(id2label),
                                                    id2label=id2label,
                                                    label2id=label2id)'''
    model.to(device)
    predictions,lab,pred = Test(model,testing_loader,device,id2label)
    print(classification_report([lab], [pred]))
    #print((predictions[:2]))
    
    #print(len(predictions[2][0]))
    preds_list, _ = align_predictions(predictions, lab_ids,id2label)
    categorize(preds_list)

if __name__ == "__main__":
    main()
