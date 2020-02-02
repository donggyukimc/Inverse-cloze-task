import os
import gc
import math
import random
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer
import transformers

import util
import model as Model


def check_answer(sim, contexts, answer, valid_rank=5) :
    "get accuracy based on given similarity score"
    sim = np.flip(np.argsort(sim, axis=1), axis=1)[:, :valid_rank]
    hits = []
    for a, s in zip(answer, sim) :
        hit = []
        for i in s :
            hit.append((a in contexts[i]))
        hits.append(hit)
    hits = np.array(hits)
    true_hit = np.zeros(hits.shape[0])!=0
    hit_rates = []
    for i in range(valid_rank) :
        true_hit = (hits[:, i].reshape(-1))|true_hit
        hit_rates.append(round((np.sum(true_hit)/len(true_hit))*100, 2))
        print("{} rank : {}".format(i+1, hit_rates[-1]))
    print('')
    return hit_rates[0]


def main(config) :

    # prepare data
    tokenizer = transformers.tokenization_bert.BertTokenizer.from_pretrained(config.bert_model)
    contexts, _, valid_qa = util.load_data(config, tokenizer)
    context_text = [context["clean_context"] for context in contexts]
    q_tokenized = [' '.join(qa["tokenized"]) for qa in valid_qa]
    q_wordpiece = [qa["wordpiece"] for qa in valid_qa]
    q_answer = [qa["answer"] for qa in valid_qa]

    tfidf = TfidfVectorizer(analyzer=str.split
                            , encoding="utf-8"
                            , stop_words="english"
                            , ngram_range=(1, config.ngram))

    # define TF-IDF 
    print("TF-IDF Retrieval")
    tfidf_context = tfidf.fit_transform([' '.join(context["tokenized"]) for context in contexts])
    tfidf_question = tfidf.transform(q_tokenized)
    tfidf_sim = util.get_sim(tfidf_question, tfidf_context)
    check_answer(tfidf_sim, context_text, q_answer)
    del tfidf_context
    del tfidf_question
    gc.collect()

    # define ICT model
    config.devices = [int(device) for device in config.devices.split('_')]
    if config.use_cuda :
        config.device = config.devices[0]
    else :
        config.device = "cpu"
    vocab = dict()
    for k, v in tokenizer.vocab.items() :
        vocab[k] = v
    start_token = vocab["[CLS]"]
    model = Model.Encoder(config)
    if config.use_cuda :
        model.cuda()
        model = nn.DataParallel(model, device_ids=config.devices)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss = nn.CrossEntropyLoss()

    # make data loader
    def get_loader(data, batch_size) :
        data = TensorDataset(torch.from_numpy(data))
        return DataLoader(data
            , batch_size=batch_size
            , shuffle=True
            , sampler=None, drop_last=True)
    loader = get_loader(np.array([i for i in range(len(contexts))]), config.batch_size)

    def get_batch(index, contexts, start_token) :
        "make ICT batch data"
        sentence = [contexts[i]["sentence"] for i in index] # get sentences of paragraphs
        target_sentence = [random.randint(0, len(sen)-1) for sen in sentence] # set target sentence for ICT training
        remove_target = [random.random()<(1-config.remove_percent) for _ in range(len(target_sentence))] # determine removal of original sentence as mention in paper
        target_context = [sen[:i]+sen[i+remove:] for i, sen, remove in zip(target_sentence, sentence, remove_target)] # set sentences of target context
        target_context = [[y for x in context for y in x] for context in target_context] # concat sentences of context
        target_context = [[start_token]+context for context in target_context]
        target_sentence = [sen[i] for i, sen in zip(target_sentence, sentence)]
        target_sentence = [[start_token]+sen for sen in target_sentence]
        s, s_mask = util.pad_sequence(target_sentence, max_seq=config.max_seq, device=config.device) # pad sequence
        c, c_mask = util.pad_sequence(target_context, max_seq=config.max_seq, device=config.device)
        return s, s_mask, c, c_mask

    def save(model, epoch, accuracy) :
        "save model weight"
        model_to_save = model.module if hasattr(model,
                        'module') else model 
        save_dict = {
            'epoch' : epoch
            ,'accuracy' : accuracy
            ,'model': model_to_save.state_dict()
        }
        torch.save(save_dict, config.model_weight)

    def load(model, device) :
        "load model weight"
        model_to_load = model.module if hasattr(model,
                        'module') else model 
        load_dict = torch.load(config.model_weight
                                    , map_location=lambda storage
                                    , loc: storage.cuda(device))
        model_to_load.load_state_dict(load_dict['model'])
        return model_to_load

    def get_semantic_sim(model) :
        "make semantic embedding of context, question. and get similarity"
        context_embedding = []
        question_embedding = []
        model.eval()
        with torch.no_grad() :
            for i in tqdm(range(0, len(contexts), config.test_batch_size)) :
                c = [[y for x in context["sentence"] for y in x] for context in contexts[i:i+config.test_batch_size]]
                c, c_mask = util.pad_sequence(c, max_seq=config.max_seq, device=config.device)
                c_encode = model(x=c, x_mask=c_mask)
                context_embedding.append(c_encode.detach().cpu().numpy())
            for i in tqdm(range(0, len(q_wordpiece), config.test_batch_size)) :
                q = [tokens for tokens in q_wordpiece[i:i+config.test_batch_size]]
                q, q_mask = util.pad_sequence(q, max_seq=config.max_seq, device=config.device)
                q_encode = model(x=q, x_mask=q_mask)
                question_embedding.append(q_encode.detach().cpu().numpy())
        context_embedding = np.concatenate(context_embedding, axis=0)
        question_embedding = np.concatenate(question_embedding, axis=0)
        return util.get_sim(question_embedding, context_embedding)  

    # train ICT model
    max_accuracy = -math.inf
    print("ICT model Retrieval.")
    for e in range(config.epoch) :
        model.train()
        avg_loss = .0
        batch_num = len(loader)
        for batch in tqdm(loader, total=batch_num) :
            batch = batch[0]
            s, s_mask, c, c_mask = get_batch(batch, contexts, start_token)
            s_encode = model(x=s, x_mask=s_mask)
            c_encode = model(x=c, x_mask=c_mask)
            logit = torch.matmul(s_encode, c_encode.transpose(-2, -1))
            target = torch.from_numpy(np.array([i for i in range(batch.size(0))])).long().to(config.device)
            loss_val = loss(logit, target).mean()
            avg_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("{} epoch, train loss : {}".format(e+1, round(avg_loss/batch_num, 2)))

        semantic_sim = get_semantic_sim(model)
        accuracy = check_answer(semantic_sim, context_text, q_answer)
        if accuracy > max_accuracy :
            max_accuracy = accuracy
            save(model, e+1, accuracy)

    # evaluate model with best performance weight
    model = load(model, config.device)
    semantic_sim = get_semantic_sim(model)
    check_answer(semantic_sim, context_text, q_answer)

    # evalute ensemble 
    check_answer(semantic_sim*(1-config.sim_ratio)+tfidf_sim*config.sim_ratio
                    , context_text, q_answer)


if __name__=="__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file", default="data.pkl"
                        , type=str, help="filename to save data")
    parser.add_argument("--model_weight", default="best.w"
                        , type=str, help="filename of model weight")

    parser.add_argument("--ngram", default=2, type=int)
    parser.add_argument("--valid_rank", default=5, type=int)

    parser.add_argument("--do_lower", default=True, type=bool)
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--max_seq", default=256, type=int)

    parser.add_argument("--epoch", default=20, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--remove_percent", default=1e-1, type=float)

    parser.add_argument("--sim_ratio", default=0.1, type=float)

    parser.add_argument("--use_cuda", default=True)
    parser.add_argument("--devices"
                        , type=str
                        , default='0_1_2_3'
                        , help="gpu device ids to use concatend with '_' ex.'0_1_2_3'")

    main(parser.parse_args())