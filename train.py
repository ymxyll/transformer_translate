import torch
import torch.nn as nn
from tqdm import tqdm
from transformer_model import Transformer
from data_load import load_train_data, load_test_data, load_cn_vocab, load_en_vocab, get_batch_indices


def main(device='cuda'):
    # 英译汉任务
    Y_train, X_train = load_train_data()
    Y_test, X_test = load_test_data()
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()
    model = Transformer(
        src_vocab_size=len(en2idx),
        tgt_vocab_size=len(cn2idx),
        ).to(device)

    lr = 1e-3
    PAD_ID = 0
    epochs = 20
    batch_size = 4
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_func = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    highest_acc, batch_acc  = 0, 0
    for i in tqdm(range(epochs), desc='Train '):
        for index, cur_idx in get_batch_indices(len(X_train), batch_size):
            x_batch = torch.LongTensor(X_train[index]).to(device)
            y_batch = torch.LongTensor(Y_train[index]).to(device)
            y_hat = model(x_batch, y_batch)
            preds = torch.argmax(y_hat, -1)
            y_label_mask = y_batch != PAD_ID
            correct = preds == y_batch
            # 防止pad对最终值产生影响
            acc = torch.sum(y_label_mask * correct) / torch.sum(y_label_mask)
            n, seq_len = y_batch.shape
            y_hat = y_hat.view(n*seq_len, -1)
            y_batch = y_batch.view(n*seq_len)
            loss = loss_func(y_hat, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters, 1) #梯度加一个缩放项感觉跟改变学习率没啥区别
            optimizer.step()
            if cur_idx % 100 == 0:
                print(f' loss: {loss.item()} acc: {acc.item()}')
            if cur_idx == (len(X_train)//batch_size):
                batch_acc = acc
        if batch_acc > highest_acc:
            highest_acc = batch_acc
            model_path = f'model/epoch{i}acc{highest_acc}.pth'
            torch.save(model.state_dict(), model_path)
        
if __name__ == '__main__':
    main()
