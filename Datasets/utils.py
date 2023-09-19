import torch
from torch.utils.data import Dataset
from itertools import chain
import numpy as np

#split validation data
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


#truncate and pad
def padding_seq(dictList):
    for i in range(0,len(dictList)):
        if len(dictList[i])>15:
            max_len=len(dictList[i])
            dictList[i]=dictList[i][-15:]
        if len(dictList[i])<15:
            w=15-len(dictList[i])
            dictList[i]=['padding_id']*w+dictList[i]
        # dictList[i]= ['CLS']+dictList[i]

    return dictList

#convert item_ids from 1 (o for padding ID)
def type_conv(data, convert_dict):
    data_new=[]
    for each in data:
        for i in range(0,len(each)):
            try:
                each[i]= convert_dict[str(each[i])]
            except:
                each[i]=0
        data_new.append(each)
    return data_new


def Prepare_Data(all_train, train, test, transe_emb):
    
    train_seq=train[0]
    train_y=train[1]

    test_seq=test[0]
    test_y=test[1]

    vocab = list(set(chain(*all_train)))
    vocab = [str(each) for each in vocab]

    train_x=padding_seq(train_seq)
    test_x=padding_seq(test_seq)

    #Transform the data
    convert_dict=dict()
    convert_dict["padding_id"]=0
    i=1
    for each in vocab:
        convert_dict[each]=i
        i=i+1

    train_x= type_conv(train_x, convert_dict)
    test_x= type_conv(test_x, convert_dict)
    
    train_x=np.array(train_x)
    test_x=np.array(test_x)

    train_x = train_x.astype(int)
    test_x = test_x.astype(int)

    train_y= [convert_dict[str(each)] for each in train_y]
    test_y1=[]
    for each in test_y:
        try:
            test_y1.append(convert_dict[str(each)])
        except:
            test_y1.append(0)
    print(" data conversion done")
    embeddings =  create_emb_matrix(vocab, transe_emb, convert_dict)
    return (train_x, train_y), (test_x, test_y1), embeddings, vocab


def create_emb_matrix(vocab, transe_emb, convert_dict):
    embed_dim =200
    total_vocab = len(vocab)+1
    unknown_id=np.zeros((200,))

    #create embedding matrix
    vocab = list(vocab)
    embedding_matrix = np.zeros((total_vocab, embed_dim))
    embedding_matrix[0] = unknown_id

    for each in vocab:
        embedding_matrix[convert_dict[each]] =transe_emb[str(each)]


    embeddings = torch.from_numpy(embedding_matrix)
    embeddings=embeddings.to(dtype= torch.float32) 
    return embeddings


class RecSysDataset(Dataset):
    
    """define the pytorch Dataset class for yoochoose and diginetica datasets.
    """
    def __init__(self, data):
        self.data = data
        print('-'*50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-'*50)
        
    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        return session_items, target_item

    def __len__(self):
        return len(self.data[0])





