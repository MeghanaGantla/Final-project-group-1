import sklearn
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

import os

current_dir = os.getcwd()
path = os.path.join(current_dir, 'Data')
data_path = os.path.join(path, 'preprocessed_data.csv')
df = pd.read_csv(data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {}.\n".format(device))

df['text'] = df['clean_text']
df.drop(['subject', 'full_text', 'clean_text'], axis=1, inplace=True)
# drop null values in the data
df.dropna(inplace=True)

training, sub = train_test_split(df, stratify=df.target.values, random_state=6312, test_size=0.2, shuffle=True)
validation, test = train_test_split(sub, stratify=sub.target.values, random_state=6312, test_size=0.25, shuffle=True)

training.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# distilbert
num_labels = 2
batch_size = 32
max_len = 256
num_epoch = 3
lr = 1e-5
check = "roberta-base"
tokenize = RobertaTokenizer.from_pretrained(check, do_lower_case=True)

train = tokenize(list(training.text.values), padding=True, max_length=max_len, truncation=True)
t_input = train['input_ids']
t_torch_inputs = torch.tensor(t_input)
t_mask = train['attention_mask']
t_torch_mask = torch.tensor(t_mask)
t_torch_labels = torch.tensor(training.target.values)
t_data = TensorDataset(t_torch_inputs, t_torch_mask, t_torch_labels)
t_sampler = RandomSampler(t_data)
t_dataloader = DataLoader(t_data, sampler=t_sampler, batch_size=batch_size)

validate = tokenize(list(validation.text.values), truncation=True, padding=True, max_length=max_len)
v_input = validate['input_ids']
v_torch_inputs = torch.tensor(v_input)
v_masks = validate['attention_mask']
v_torch_masks = torch.tensor(v_masks)
v_torch_labels = torch.tensor(validation.target.values)
v_data = TensorDataset(v_torch_inputs, v_torch_masks, v_torch_labels)
v_sampler = SequentialSampler(v_data)
v_dataloader = DataLoader(v_data, sampler=v_sampler, batch_size=batch_size)


def parameter_count(given_model):
    return sum(p.numel() for p in given_model.parameters() if p.requires_grad)


def metric_evaluation(predicts, labels):
    predictions = predicts.argmax(axis=1, keepdim=True)
    accuracy = accuracy_score(y_true=labels.to('cpu').tolist(), y_pred=predictions.detach().cpu().numpy())
    return accuracy


def training(model, dataloader, optimizer, device, scheduler):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    for data in dataloader:
        inputs, masks, labels = [d.to(device) for d in data]
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        train_acc += metric_evaluation(outputs.logits, labels)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def validate(model, dataloader, device):
    model.eval()
    valid_loss, valid_acc = 0.0, 0.0

    with torch.no_grad():
        for data in dataloader:
            inputs, masks, labels = [d.to(device) for d in data]
            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            valid_loss += loss.item()
            logits = outputs.logits
            valid_acc += metric_evaluation(logits, labels)

    valid_loss /= len(dataloader)
    valid_acc /= len(dataloader)

    return valid_loss, valid_acc


model = RobertaForSequenceClassification.from_pretrained(check, num_labels=num_labels, output_hidden_states=False,
                                                            output_attentions=False)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=lr)
total_steps = len(t_dataloader) * num_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
num_params = parameter_count(model)
print("trainable parameters: {}".format(num_params))


train_losses, validation_losses, train_accuracies, validation_accuracies = [], [], [], []
best_loss = float('inf')

for epoch in range(num_epoch):
    epoch_t_loss, epoch_t_acc = training(model, t_dataloader, optimizer, device, scheduler)
    epoch_v_loss, epoch_v_acc = validate(model, v_dataloader, device)
    train_losses.append(epoch_t_loss)
    validation_losses.append(epoch_v_loss)
    train_accuracies.append(epoch_t_acc)
    validation_accuracies.append(epoch_v_acc)
    if epoch_v_loss < best_loss:
        best_loss = epoch_v_loss
        torch.save(model.state_dict(), 'model.pt')

    print(f"Epoch: {epoch}, Train Loss: {epoch_t_loss:.4f}, Train Accuracy: {epoch_t_acc:.2f}%")
    print(f"Epoch: {epoch}, Validation Loss: {epoch_v_loss:.4f}, Validation Accuracy: {epoch_v_acc:.2f}%\n")


# Testing
model.load_state_dict(torch.load('model.pt'))
test = tokenize(list(test[:1000].text.values), max_length=30, truncation=True, padding=True)
test_input = test['input_ids']
test_tensor_input = torch.tensor(test_input)
test_masks = test['attention_mask']
test_torch_masks = torch.tensor(test_masks)

with torch.no_grad():
    test_tensor_input = test_tensor_input.to(device)
    test_masks = test_masks.to(device)
    outputs = model(test_tensor_input, test_masks)
    logits = outputs.logits
    batch_logits = logits.detach().cpu().numpy()
    predict = np.argmax(batch_logits, axis=1)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
axes[0, 0].plot(np.arange(1, 4), validation_losses)
axes[0, 0].set_title('Loss vs Epochs - Validation')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Loss')
axes[0, 1].plot(np.arange(1, 4), train_losses)
axes[0, 1].set_title('Loss vs Epochs - Train')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('Loss')
axes[1, 0].plot(np.arange(1, 4), train_accuracies)
axes[1, 0].set_title('Accuracy vs Epochs - Train')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 1].plot(np.arange(1, 4), validation_accuracies)
axes[1, 1].set_title('Accuracy vs Epochs - Validation')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].set_ylabel('Accuracy')
plt.tight_layout()
plt.show()
print("ROC AUC Score: {}".format(roc_auc_score(y_true=test[:1000].target.values, y_score=predict)))
print('f1-score: ', sklearn.metrics.f1_score(test[:1000].target.values, predict))
print(classification_report(test[:1000].target.values, predict))
