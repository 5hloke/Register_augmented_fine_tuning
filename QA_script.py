import json
import random
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer, BertConfig

from model import RegBertForQA

from tqdm.auto import tqdm
import numpy as np
import evaluate
import collections
from collections import Counter
from nltk.util import ngrams
from scipy.spatial.distance import jensenshannon
import warnings
import gc
warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') ## 

    


S_lang2file = {
    'en' : 'TyDiQA English Data/tydiqa.en.train.json'
}

T_lang2file = {
    'en' : 'TyDiQA English Data/tydiqa.en.dev.json'
}

accuracy_dict = {} # for storing all the test accuracies in the form { (S,T,SHOT) , Acc }
init_acc_dict = {}

# All Languages: en, fi, ar, bn, id, ko, ru, sw, te = 9
# Total Language pairs = 9*9 = 81

SHOT = 0

max_length = 434
stride = 128

path = ""

metric = evaluate.load("squad")


def read_data(path):  
    with open(path, 'rb') as f:
        squad = json.load(f)
    contexts = []
    questions = []
    answers = []
    id = []
    for group in squad['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
                    id.append(qa['id'])
    return contexts, questions, answers, id

def get_s_data(S,T,SHOT):
    s_path = path + S_lang2file[S]
    s_context, s_q, s_a, s_i = read_data(s_path)
    s_tydi = []
    for _ in range(len(s_a)):
        s_tydi.append({})
        s_tydi[_]['answers'] = s_a[_]
        s_tydi[_]['context'] = s_context[_]
        s_tydi[_]['question'] = s_q[_]
        s_tydi[_]['id'] = s_i[_]
    if SHOT>0:
        few_shot_path = path + S_lang2file[T]
        fs_context, fs_q, fs_a, fs_i = read_data(few_shot_path)
        for _ in range(SHOT):
            s_tydi.append({})
            s_tydi[len(s_tydi) - 1]['answers'] = fs_a[_]
            s_tydi[len(s_tydi) - 1]['context'] = fs_context[_]
            s_tydi[len(s_tydi) - 1]['question'] = fs_q[_]
            s_tydi[len(s_tydi) - 1]['id'] = fs_i[_]
    s_data = Dataset.from_list(s_tydi)
    return s_data

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"]
        end_char = answer["answer_start"] + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)
        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def get_t_data(T):
    t_path = path + T_lang2file[T]
    t_context, t_q, t_a, t_i = read_data(t_path)
    t_tydi = []
    for _ in range(len(t_a)):
        t_tydi.append({})
        t_tydi[_]['answers'] = t_a[_]
        t_tydi[_]['context'] = t_context[_]
        t_tydi[_]['question'] = t_q[_] 
        t_tydi[_]['id'] = t_i[_] 
    t_data = Dataset.from_list(t_tydi)
    return t_data


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    inputs["example_id"] = example_ids
    return inputs


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []
        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            start_indexes = np.argsort(start_logit)[-1 : -20 - 1 : -1].tolist() #n_best
            end_indexes = np.argsort(end_logit)[-1 : -20 - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > 60 #max_answer_len
                    ):
                        continue
                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)
        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})
    theoretical_answers = [{"id": ex["id"], "answers": {"text":[ex["answers"]["text"]], "answer_start":[ex["answers"]["answer_start"]]}} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)



def model_train(tr_data, te_data, num_registers, config = None, model = None):
    data_collator = DefaultDataCollator()

    print('from model_train(), num_reg=', num_registers)

    print('model.bert.num_registers=', model.bert.num_registers)
    
    training_args = TrainingArguments(
        output_dir='QA_OP',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tr_data,
        eval_dataset=te_data,
        data_collator=data_collator,
    )
    trainer.train()
    return trainer



### MODEL TRAINING
config = BertConfig.from_pretrained("bert-base-uncased")
for num_registers in range(20, 101, 5):
    model = RegBertForQA(config=config, num_registers=num_registers)
    for S in S_lang2file.keys(): # S_lang2file.keys()
        train_counter = 1
        for T in T_lang2file.keys(): # T_lang2file.keys()
            s_data = get_s_data(S,T,SHOT)
            train_dataset = s_data.map(preprocess_training_examples, batched=True, remove_columns=s_data.column_names)
            t_data = get_t_data(T)
            validation_dataset = t_data.map(preprocess_validation_examples, batched=True, remove_columns=t_data.column_names)
            if train_counter == 1:
                print('from train, num_reg=', num_registers)
                # save_path = f'model_num_reg_{num_registers}.pth'
                trainer = model_train(train_dataset, validation_dataset, num_registers=num_registers, model = model)
                # torch.save(trainer.model.state_dict(), save_path)
                trainer.save_model(f'model_num_reg_{num_registers}')
                tokenizer.save_pretrained(f'tokenizer_num_reg_{num_registers}')
                print('model saved for num_reg = ', num_registers)
                
            train_counter = train_counter + 1
            predictions, _, _ = trainer.predict(validation_dataset)
            start_logits, end_logits = predictions
            print('start_logits Pred: ',start_logits.shape)
            f1 = compute_metrics(start_logits, end_logits, validation_dataset, t_data)
            accuracy_dict[(S,T,SHOT)] = f1

            file_name = f'Acc_QA_num_reg{num_registers}'
            with open(file_name, "w") as fp:
                print(accuracy_dict, file=fp)
            print('accuracy saved for num_reg=', num_registers)
            gc.collect()
            torch.cuda.empty_cache()

###
