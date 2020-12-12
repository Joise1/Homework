# coding=utf-8
import torch
import os
import datetime
import unicodedata


class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask, sent=None, trigger=None, event_type=None):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask
        self.sent = sent
        self.trigger = trigger
        self.event_type = event_type
        self.p = 0


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def read_corpus_tri_cls(path, max_length, label_dic, vocab):
    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    result = []
    for line in content:
        sentence_a, sentence_b, label = line.strip().split('|||')
        tokens_a = sentence_a.split()
        tokens_b = sentence_b.split()
        if len(tokens_a) + len(tokens_b) > max_length-3:
            tokens_b = tokens_b[0:(max_length-3-len(tokens_a))]
        tokens_f = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        # see
        tokens_f = [list(vocab.keys())[int(id)] for id in input_ids]
        label = int(label)
        input_mask = [1] * len(input_ids)
        type_mask = [0] * (2+len(tokens_a)) + [1] * (max_length-2-len(tokens_a))
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
        result.append((input_ids, input_mask, type_mask, label))
    return result


def read_corpus(path, max_length, label_dic, vocab, content=None):
    """
    :param path:数据文件路径
    :param max_length: 最大长度
    :param label_dic: 标签字典
    :return:
    """
    if not content:
        file = open(path, encoding='utf-8')
        content = file.readlines()
        file.close()
    result = []
    for line in content:
        text, label = line.strip().split('|||')
        tokens = text.split()
        label = label.split()
        if len(tokens) > max_length-2:
            tokens = tokens[0:(max_length-2)]
            label = label[0:(max_length-2)]
        tokens_f = ['[CLS]'] + tokens + ['[SEP]']
        label_f = ["<start>"] + label + ['<eos>']
        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        label_ids = [label_dic[i] for i in label_f]
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_dic['<pad>'])
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(label_ids) == max_length
        # get trigger
        triggers = []
        trigger = {'begin_pos': 0, 'end_pos': 0}
        for idx, l in enumerate(label):
            if l == 'B-TRI':
                trigger['begin_pos'] = idx
                if idx < len(label) - 1 and label[idx + 1] == 'O':
                    trigger['end_pos'] = idx
                    triggers.append(trigger)
                    trigger = {'begin_pos': 0, 'end_pos': 0}
            elif l == 'I-TRI':
                trigger['end_pos'] = idx
                if idx < len(label) - 1 and label[idx + 1] == 'O':
                    triggers.append(trigger)
                    trigger = {'begin_pos': 0, 'end_pos': 0}
        feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids,
                                sent=tokens, trigger=triggers)
        result.append(feature)
    return result


def save_model(model, epoch, path='result', **kwargs):
    """
    默认保留所有模型
    :param model: 模型
    :param path: 保存路径
    :param loss: 校验损失
    :param last_loss: 最佳epoch损失
    :param kwargs: every_epoch or best_epoch
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if kwargs.get('name', None) is None:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
        name = cur_time + '--epoch:{}'.format(epoch)
    else:
        name = kwargs.get('name', None)
        full_name = os.path.join(path, name)
        torch.save(model.state_dict(), full_name)
        print('Saved model at epoch {} successfully'.format(epoch))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write(name)
            print('Write to checkpoint')


def load_model(model, path='result', **kwargs):
    if kwargs.get('name', None) is None:
        with open('{}/checkpoint'.format(path)) as file:
            content = file.read().strip()
            name = os.path.join(path, content)
    else:
        name=kwargs['name']
        name = os.path.join(path, name)
    model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model


