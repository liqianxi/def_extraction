import torch

import time
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torchtext.data.functional import to_map_style_dataset
from model import *
from utils import *
from torch.utils.data import DataLoader
from def_keyword_extraction import *
from transformers import *


DEFINITION_MODEL_PATH = "./linear_model.pt"

def make_prediction(text, filename):
    result_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters
    EPOCHS = 10 # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 64 # batch size for training
    
    ag_news_label = {1: "Definition",
                    0:"No_definition"}
    tokenizer = get_tokenizer('basic_english')
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: x

    def predict(text, text_pipeline):
        with torch.no_grad():
            text = torch.tensor(text_pipeline(text))
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item()

    model = torch.load(DEFINITION_MODEL_PATH)
    model.eval()
    vocab = torch.load('vocab_obj.pth')
    model = model.to("cpu")
    
    read_out = text.replace('etc.',"").replace("e.g.","")
    raw = [i for i in read_out.splitlines() if i]
    total = []

    for each in raw:
        total += [i for i in each.split('.') if i]
    
    for each in total:
        result = predict(each, text_pipeline)
        
        if result:
            keyword_generated = extract_key_term_from_sentence(each)
            if keyword_generated:
                result_list.append((keyword_generated,each,filename))

    return result_list

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters
    EPOCHS = 10 # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 64 # batch size for training
    PATH = "./linear_model.pt"
    ag_news_label = {1: "Definition",
                    0:"No_definition"}
    tokenizer = get_tokenizer('basic_english')
    #print("vocab",type(vocab))
    #text_pipeline = lambda x: vocab(tokenizer(x))
    def text_pipeline(x):
        print(type(vocab))
        return vocab(tokenizer(x))
    
    label_pipeline = lambda x: x



    def predict(text, text_pipeline):
        with torch.no_grad():
            text = torch.tensor(text_pipeline(text))
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item()

    train_folder, test_folder, dev_folder = './train', './test','./dev'
    train_set = dataset_convert(train_folder)  
    test_set = dataset_convert(test_folder)  
    dev_set = dataset_convert(dev_folder)


    train_iter = iter(train_set)






    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    print("vocab2",type(vocab))
    vocab.set_default_index(vocab["<unk>"])
    train_iter = iter(train_set)
    test_iter = iter(test_set)
    dev_iter = iter(dev_set)







    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)



    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                    '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                total_acc/total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(dataloader):
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                loss = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc/total_count


    # Hyperparameters
    EPOCHS = 10 # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 64 # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None


    train_dataset = to_map_style_dataset(train_set)
    test_dataset = to_map_style_dataset(test_set)
    dev_dataset = to_map_style_dataset(dev_set)



    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_batch)


    if os.path.isfile(PATH):


        model = torch.load(PATH)
        model.eval()

        if model:
            ex_text_str = "A house is a building that humans live in"
            model = model.to("cpu")
            print("This is a %s " %ag_news_label[predict(ex_text_str, text_pipeline)])
            exit()


    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        accu_val = evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                'valid accuracy {:8.3f} '.format(epoch,
                                                time.time() - epoch_start_time,
                                                accu_val))
            print('-' * 59)
    torch.save(vocab, 'vocab_obj.pth')
    torch.save(model, PATH)
    print('Checking the results of test dataset.')
    accu_test = evaluate(test_dataloader)
    print('test accuracy {:8.3f}'.format(accu_test))




    ex_text_str = "A house is a building that humans live in"


    model = model.to("cpu")
    print("This is a %s " %ag_news_label[predict(ex_text_str, text_pipeline)])
if os.path.isfile(DEFINITION_MODEL_PATH):
    res = make_prediction("A house is a building that humans live in.A cat is a building that humans live in.","1.txt")
    print(res)
else:
    train()