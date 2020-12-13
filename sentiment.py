import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DP
from torch.optim.lr_scheduler import ExponentialLR
from transformers import (AdamW, AlbertForMaskedLM, AutoModel, AutoTokenizer,
                          BertTokenizer)

from elm import classic_ELM


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class ELM_Classifier_finetune:
    def __init__(self, args) -> None:
        """Use ELM with fintuned language model for sentiment classification

        Args:
            args (dict): contain all the arguments needed.
                - model_name(str): the name of the transformer model
                - bsz(int): batch size
                - epoch: epochs to train
                - type(str): fintuned type
                  - base: train only ELM
                  - finetune_elm: train transformers with ELM directly
                  - finetune_classifier: train transformers with classifier
                  - finetune_classifier_elm: train transformers with classifier,
                    and use elm replace the classifier
                  - finetune_classifier_beta: train transformers with classifier,
                    and use pinv to calculate beta in classifier
                - learning_rate(float): learning_rate for finetuning
        """
        # load configuration
        self.model_name = args.get('model_name', 'bert-base-uncased')
        self.bsz =  args.get('batch_size', 10)
        self.epoch = args.get('epoch_num', 2)
        self.learning_rate = args.get('learning_rate', 0.001)
        self.training_type = args.get('training_type', 'base')
        self.debug = args.get('debug', True)
        self.eval_epoch = args.get('eval_epoch', 1)
        self.lr_decay = args.get('learning_rate_decay', 0.99)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device
        self.n_gpu = torch.cuda.device_count()

        # load pretrained model
        if (self.model_name == 'bert-base-uncased') or \
                (self.model_name == 'distilbert-base-uncased') or \
                (self.model_name == 'albert-base-v2'):
            self.pretrained_model = AutoModel.from_pretrained(self.model_name)
            self.pretrained_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name)
            input_shape = 768
            output_shape = 256
        elif (self.model_name == 'prajjwal1/bert-tiny'):
            self.pretrained_model = AutoModel.from_pretrained(self.model_name)
            self.pretrained_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, model_max_length=512)
            input_shape = 128
            output_shape = 64
        elif self.model_name == 'voidful/albert_chinese_xxlarge':
            self.pretrained_model = AlbertForMaskedLM.from_pretrained(
                self.model_name)
            self.pretrained_tokenizer = BertTokenizer.from_pretrained(
                self.model_name)
            input_shape = 768
            output_shape = 256
        else:
            raise TypeError("Unsupported model name")
        self.pretrained_model.to(device)
        device_ids = None
        if self.n_gpu > 1:
            device_ids=range(torch.cuda.device_count())
            self.pretrained_model = DP(self.pretrained_model, device_ids=device_ids)

        # load specific model
        if (self.training_type == 'finetune_classifier') or \
            (self.training_type == 'finetune_classifier_elm'):
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(input_shape, 2)
            )
            self.loss_func = torch.nn.CrossEntropyLoss()
            self.classifier.to(device)
            if self.n_gpu > 1:
                self.classifier = DP(self.classifier, device_ids=device_ids)
        if (self.training_type == 'base') or \
            (self.training_type =='finetune_classifier_elm'):
            self.elm = classic_ELM(input_shape, output_shape)
        if (self.training_type == 'finetune_classifier_linear'):
            self.elm = classic_ELM(None, None)
            self.classifier = torch.nn.Sequential(OrderedDict([
                ('w', torch.nn.Linear(input_shape, output_shape)),
                ('act', torch.nn.Sigmoid()),
                ('beta', torch.nn.Linear(output_shape, 2)),
            ]))
            self.loss_func = torch.nn.CrossEntropyLoss()
            self.classifier.to(device)
            if self.n_gpu > 1:
                self.classifier = DP(self.classifier, device_ids=device_ids)

        # load processor, trainer, evaluator, inferer.
        processors = {
            'base': self.__processor_base__,
            'finetune_classifier': self.__processor_base__,
            'finetune_classifier_elm': self.__processor_base__,
            'finetune_classifier_linear': self.__processor_base__,
        }
        trainers = {
            'base': self.__train_base__,
            'finetune_classifier': self.__train_finetune_classifier__,
            'finetune_classifier_elm': self.__train_finetune_classifier_elm__,
            'finetune_classifier_linear': self.__train_finetune_classifier_linear__,
        }
        evaluators = {
            'base': self.__eval_base__,
            'finetune_classifier': self.__eval_finetune_classifier__,
            'finetune_classifier_elm': self.__eval_base__,
            'finetune_classifier_linear': self.__eval_finetune_classifier_linear__,
        }
        inferers = {
            'base': self.__infer_base__,
            'finetune_classifier': self.__infer_finetune_classifier__,
            'finetune_classifier_elm': self.__infer_finetune_classifier_elm__,
            'finetune_classifier_linear': self.__infer_base__
        }
        self.processor = processors[self.training_type]
        self.trainer = trainers[self.training_type]
        self.evaluator = evaluators[self.training_type]
        self.inferer = inferers[self.training_type]


    def preprocess(self, *list_arg, **dict_arg):
        """
        Unified preprocess
        """
        print('Preprocessing......')
        return self.processor(*list_arg, **dict_arg)

    def train(self, *list_arg, **dict_arg):
        """
        Unified train
        """
        print('Training......')
        acc = self.trainer(*list_arg, **dict_arg)
        print('Best Accuracy:', acc)
        return acc

    def eval(self, *list_arg, **dict_arg):
        """
        Unified evalate
        """
        print('Evaluating......')
        return self.evaluator(*list_arg, **dict_arg)

    def infer(self, *list_arg, **dict_arg):
        """
        Unified inference
        """
        print('Infering......')
        return self.inferer(*list_arg, **dict_arg)

    def __train_base__(self, train_dataset, test_dataset, do_eval=True):
        # prepare to train
        self.pretrained_model.eval()
        batch_num = math.ceil(len(train_dataset.labels) / self.bsz)
        test_loader = DataLoader(train_dataset, batch_size=self.bsz, shuffle=True)
        collect_out = []
        collect_label = []
        
        # collect outputs and train elm
        print('collecting outputs......')
        pbar = tqdm(range(batch_num))
        for batch in test_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            with torch.no_grad():
                outputs = self.pretrained_model(
                    input_ids, attention_mask=attention_mask)
                pooler = outputs[1]
                collect_out.append(pooler.cpu().numpy())
                collect_label.append(labels.cpu().numpy())
            pbar.update()
        pbar.close()

        # train elm
        print('Train ELM......')
        collect_out = np.array(collect_out)
        collect_label = np.array(collect_label)
        num, bsz, hidden_dim = collect_out.shape
        collect_out = collect_out.reshape(num*bsz, hidden_dim)
        collect_label = collect_label.reshape(num*bsz)
        self.elm.train(collect_out, collect_label)

        # evaluate
        acc = 0
        if do_eval:
            acc = self.eval(test_dataset)
        return acc

    def __train_finetune_classifier__(self, train_dataset, test_dataset, do_eval=True):
        # set train mode
        self.pretrained_model.train()
        self.classifier.train()
        
        # prepare optimizer
        batch_num = math.ceil(len(train_dataset.labels) / self.bsz)
        train_loader = DataLoader(train_dataset, batch_size=self.bsz, shuffle=True)
        params = [
            {'params': self.pretrained_model.parameters()},
            {'params': self.classifier.parameters()}
        ]
        optimizer = AdamW(params, lr=self.learning_rate)
        scheduler = ExponentialLR(optimizer, self.lr_decay)
        
        # train
        best_acc = 0
        epochs = self.epoch if do_eval else 1
        for epoch in range(epochs):
            pbar = tqdm(range(batch_num))
            losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.pretrained_model(input_ids, attention_mask=attention_mask)
                pooler = outputs[1]
                outputs = self.classifier(pooler)
                loss = self.loss_func(outputs, labels)
                if self.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                pbar.update()
                losses.append(loss.data.cpu())
                descrip = 'Train epoch:%3d   Loss:%6.3f' % (epoch, loss.data.cpu())
                if not do_eval:
                    descrip = 'Loss:%6.3f' % loss.data.cpu()
                pbar.set_description(descrip)
            scheduler.step()
            # set average epoch description
            avg_loss = torch.mean(torch.tensor(losses))
            final_descrip = 'Epoch:%2d  Average Loss:%6.3f' % (epoch, avg_loss)
            if not do_eval:
                final_descrip = 'Average Loss:%6.3f' % avg_loss
            pbar.set_description(final_descrip)
            pbar.close()
            # eval for epochs
            if (epoch % self.eval_epoch == 0) and do_eval:
                test_acc = self.eval(test_dataset)
                best_acc = max(test_acc, best_acc)
                self.pretrained_model.train()
                self.classifier.train()
        return best_acc

    def __train_finetune_classifier_elm__(self, train_dataset, test_dataset, do_eval=True):
        best_acc = 0
        for epoch in range(self.epoch):
            print('Epoch %d' % epoch)
            self.__train_finetune_classifier__(train_dataset, test_dataset, do_eval=False)
            self.__train_base__(train_dataset, test_dataset, do_eval=False)
            if do_eval and (epoch % self.eval_epoch == 0):
                acc = self.eval(test_dataset)
                best_acc = max(best_acc, acc)
        return best_acc

    def __train_finetune_classifier_linear__(self, train_dataset, test_dataset, do_eval=True):
        best_acc = 0
        batch_num = math.ceil(len(train_dataset.labels) / self.bsz)
        for epoch in range(self.epoch):
            # train classifier
            print('Epoch %d' % epoch)
            self.__train_finetune_classifier__(train_dataset, test_dataset, do_eval=False)

            # calculate last layer with model_output
            print('collecting outputs......')
            collect_out = []
            collect_label = []
            self.pretrained_model.eval()
            self.classifier.eval()
            test_loader = DataLoader(train_dataset, batch_size=self.bsz, shuffle=True)
            pbar = tqdm(range(batch_num))
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                with torch.no_grad():
                    outputs = self.pretrained_model(
                        input_ids, attention_mask=attention_mask)
                    pooler = outputs[1]
                    linear = self.classifier.w(pooler)
                    linear = self.classifier.act(linear)
                    collect_out.append(linear.cpu().numpy())
                    collect_label.append(labels.cpu().numpy())
                pbar.update()
            pbar.close()

            print('Train ELM......')
            collect_out = np.array(collect_out)
            collect_label = np.array(collect_label)
            num, bsz, hidden_dim = collect_out.shape
            collect_out = collect_out.reshape(num*bsz, hidden_dim)
            collect_label = collect_label.reshape(num*bsz)
            self.elm.train(collect_out, collect_label)

            if do_eval and (epoch % self.eval_epoch == 0):
                acc = self.eval(test_dataset)
                best_acc = max(best_acc, acc)
        return best_acc

    def __eval_base__(self, test_dataset):
        # prepare eval
        self.pretrained_model.eval()
        batch_num = math.ceil(len(test_dataset.labels) / self.bsz)
        test_loader = DataLoader(test_dataset, batch_size=self.bsz, shuffle=True)
        pbar = tqdm(range(batch_num))
        
        # collect tensors
        collect_out = []
        collect_label = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            with torch.no_grad():
                outputs = self.pretrained_model(
                    input_ids, attention_mask=attention_mask)
                pooler = outputs[1]
                collect_out.append(pooler.cpu().numpy())
                collect_label.append(labels.cpu().numpy())
            pbar.update()
        pbar.close()

        # evaluate
        collect_out = np.array(collect_out)
        collect_label = np.array(collect_label)
        num, bsz, hidden_dim = collect_out.shape
        collect_out = collect_out.reshape(num*bsz, hidden_dim)
        collect_label = collect_label.reshape(num*bsz)
        pred_labels = self.elm.infer(collect_out) > 0.5
        acc = pred_labels == collect_label
        acc = np.sum(acc) / len(collect_out)
        print('Total accuracy: ', acc)
        return acc

    def __eval_finetune_classifier__(self, test_dataset):
        # set eval mode
        self.pretrained_model.eval()
        self.classifier.eval()

        # prepare eval
        batch_num = math.ceil(len(test_dataset.labels) / self.bsz)
        test_loader = DataLoader(test_dataset, batch_size=self.bsz, shuffle=True)
        pbar = tqdm(range(batch_num))
        acc_list = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            with torch.no_grad():
                outputs = self.pretrained_model(
                    input_ids, attention_mask=attention_mask)
                pooler = outputs[1]
                outputs = self.classifier(pooler)
                output_label = torch.argmax(outputs, axis=1)
            acc = output_label == labels
            acc = acc.float()
            acc = torch.sum(acc) / labels.size(0)
            acc_list.append(acc.cpu())
            pbar.update()
            descrip = 'Current Accuracy:%6.3f' % acc
            pbar.set_description(descrip)
        pbar.close()
        t_acc = np.array(acc_list).mean()
        print('Total accuracy: ', t_acc)
        return t_acc

    def __eval_finetune_classifier_linear__(self, test_dataset):
        # prepare eval
        self.pretrained_model.eval()
        batch_num = math.ceil(len(test_dataset.labels) / self.bsz)
        test_loader = DataLoader(test_dataset, batch_size=self.bsz, shuffle=True)
        pbar = tqdm(range(batch_num))

        # collect tensors
        collect_out = []
        collect_label = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            with torch.no_grad():
                outputs = self.pretrained_model(
                    input_ids, attention_mask=attention_mask)
                pooler = outputs[1]
                linear = self.classifier.w(pooler)
                linear = self.classifier.act(linear)
                collect_out.append(linear.cpu().numpy())
                collect_label.append(labels.cpu().numpy())
            pbar.update()
        pbar.close()

        # evaluate
        collect_out = np.array(collect_out)
        collect_label = np.array(collect_label)
        num, bsz, hidden_dim = collect_out.shape
        collect_out = collect_out.reshape(num*bsz, hidden_dim)
        collect_label = collect_label.reshape(num*bsz)
        pred_labels = self.elm.infer(collect_out) > 0.5
        acc = pred_labels == collect_label
        acc = np.sum(acc) / len(collect_out)
        print('Total accuracy: ', acc)
        return acc

    def __infer_base__(self, texts):
        collect_out = []
        for data in tqdm(texts):
            data = list(data)
            inputs = self.pretrained_tokenizer(data,
                                               truncation=True,
                                               padding=True,
                                               return_tensors='pt',
                                               )
            outputs = self.pretrained_model(**inputs)
            collect_out.append(outputs['pooler_output'].detach().numpy())
        collect_out = np.array(collect_out)
        label = self.elm.infer(collect_out) > 0.5
        return label

    def __infer_finetune_classifier__(self, texts):
        raise NotImplementedError

    def __infer_finetune_classifier_elm__(self, texts):
        raise NotImplementedError

    def __processor_base__(self, train_text, train_label, test_text, test_label):
        """packaging dataset use torch.Dataset

        Args:
            train_text (numpy.ndarray): (trainset_num,)
            train_label (numpy.ndarray): (trainset_num,)
            test_text (numpy.ndarray): (testset_num,)
            test_label (numpy.ndarray): (testset_num,)

        Returns:
            train_text (numpy.ndarray): (batch_num, batch_size)
            train_label (numpy.ndarray): (batch_num, batch_size)
            test_text (numpy.ndarray): (batch_num, batch_size)
            test_label (numpy.ndarray): (batch_num, batch_size)
        """

        # use only first 50 sentences
        if self.debug:
            train_text = train_text[:50]
            train_label = train_label[:50]
            test_text = test_text[:50]
            test_label = test_label[:50]

        train_text = list(train_text)
        test_text = list(test_text)
        train_encodings = self.pretrained_tokenizer(train_text, truncation=True, padding=True)
        test_encodings = self.pretrained_tokenizer(test_text, truncation=True, padding=True)
        train_dataset = IMDbDataset(train_encodings, train_label)
        test_dataset = IMDbDataset(test_encodings, test_label)

        return train_dataset, test_dataset


def load_microblog():
    pass

def load_imdb():
    """Loading imdb datasets and drop all the unsup one

    Returns:
        train_text: 
        train_label:
        test_text:
        test_label:
    """
    print('Loading dataset(IMDB)......')
    # load text file and convert and remove unsup
    dataset = pd.read_csv('./datasets/imdb_master.csv')
    dataset = dataset[(dataset['label'] == 'neg') |
                      (dataset['label'] == 'pos')]
    train_set = dataset[dataset['type'] == 'train']
    train_text = np.array(train_set['review'])
    train_label = np.array(train_set['label'])
    train_label = np.array(list(map(lambda i: 1 if i=='pos' else 0, train_label)))
    test_set = dataset[dataset['type'] == 'test']
    test_text = np.array(test_set['review'])
    test_label = np.array(test_set['label'])
    test_label = np.array(list(map(lambda i: 1 if i=='pos' else 0, test_label)))

    # shuffle and split the dataset
    # trainset
    new_arg = np.arange(0, len(train_set))
    np.random.shuffle(new_arg)
    train_text = train_text[new_arg]
    train_label = train_label[new_arg]
    # testset
    new_arg = np.arange(0, len(test_set))
    np.random.shuffle(new_arg)
    test_text = test_text[new_arg]
    test_label = test_label[new_arg]

    return train_text, train_label, test_text, test_label


def main():
    # parse arg from command line
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=None,
                        help='use debug mode')
    parser.add_argument('--training_type', type=str,
                        help='training type of the model', choices=['base',
                                                                    'finetune_classifier',
                                                                    'finetune_classifier_elm',
                                                                    'finetune_classifier_linear'])
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch size')
    parser.add_argument('--epoch_num', type=int, default=None,
                        help='epoch number')
    parser.add_argument('--model_name', type=str, default=None,
                        help='name of pretrained model', choices=['bert-base-uncased',
                                                                  'distilbert-base-uncased',
                                                                  'albert-base-v2',
                                                                  'prajjwal1/bert-tiny',
                                                                  'voidful/albert_chinese_xxlarge'])
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='initial learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=None,
                        help='learning rate decay for Exponetial LR schedular')
    parser.add_argument('--eval_epoch', type=int, default=None,
                        help='evaluate for every n epoch')
    cmd_args = parser.parse_args()

    # update default args
    args = {
        'model_name': 'albert-base-v2',
        'batch_size': 2,
        'epoch_num': 1,
        'learning_rate': 5e-5,
        'learning_rate_decay': 0.9,
        'training_type': 'finetune_classifier_linear',
        'debug': False,
        'eval_epoch': 1,
    }
    cmd_args = vars(cmd_args)
    key_l = list(cmd_args.keys())
    for key in key_l:
        if cmd_args[key] is None:
            cmd_args.pop(key)
    args.update(cmd_args)

    # train
    train_text, train_label, test_text, test_label = load_imdb()
    classifier = ELM_Classifier_finetune(args)
    train_dataset, test_dataset = classifier.preprocess(train_text, train_label, test_text, test_label)
    classifier.train(train_dataset, test_dataset)
    print('Done')

if __name__ == "__main__":
    main()
