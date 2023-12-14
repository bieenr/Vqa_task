from dataset.VQA_dataset import VQADataset
from model.VQA_model_normal import VQAModel
import torchvision.transforms as transforms
import json
import spacy
import torchtext
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/vqa_lstm_cnn_experiment_1')
train_data = json.load(open(r'vqa_train.json','r'))['data']
val_data = json.load(open(r'vqa_test.json','r'))['data']
### create vocab
eng = spacy.load("en_core_web_sm")
# def my_vocab_vqa(train_data):
def get_tokens(data_iter):
    for sample in data_iter:
        question = sample['question']
        yield [token.text for token in eng.tokenizer(question)]

vocab = build_vocab_from_iterator(
    get_tokens(train_data),
    min_freq = 2,
    specials = ['<pad>','<sos>','<eos>', '<unk>'],
    special_first = True
)
vocab.set_default_index(vocab['<unk>'])
    # return vocab

classes = set([sample['answer'] for sample in train_data])

classes_to_idx = {
    cls_name: idx for idx, cls_name in enumerate(classes)
}
idx_to_classes = {
    idx: cls_name for idx, cls_name in enumerate(classes)
}
###check val_data
val_data = [data for data in val_data if data["answer"] in classes]

## create tokenize
def tokenize(question,max_sequence_length):
    tokens = [token.text for token in eng.tokenizer(question)]
    sequence = [vocab[token] for token in tokens]
    if (len(sequence) < max_sequence_length):
        sequence += [vocab['<pad>']] * (max_sequence_length - len(sequence))
    else:
        sequence = sequence[:max_sequence_length]
    return sequence
## load dataset
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

train_dataset = VQADataset(
    train_data,
    tokenize=tokenize,
    classes_to_idx=classes_to_idx,
    transform=transform,
    root_dir=r"data_raw\scene_img_abstract_v002_binary_train2017\scene_img_abstract_v002_train2017"
    )
val_dataset = VQADataset(
    val_data,
    tokenize=tokenize,
    classes_to_idx=classes_to_idx,
    transform= transform,
    root_dir=r"data_raw\scene_img_abstract_v002_binary_val2017\scene_img_abstract_v002_val2017"
    )
train_batch_size = 32
test_batch_size = 8
train_loader = DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=test_batch_size,shuffle=False)


#### Create model
n_classes = len(classes)
img_model_name = 'resnet50'
hidden_size = 128
n_layers = 1
embedding_dim = 128
dropout_prob = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VQAModel(
    n_classes=n_classes,
    img_model_name=img_model_name,
    embedding_dim=embedding_dim,
    n_layers=n_layers,
    len_vocab=len(vocab),
    hidden_size=hidden_size,
    dropout_prob=dropout_prob
    ).to(device)

### Create evaluate
def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for image, question, labels in tqdm(dataloader):
            image, question, labels = image.to(device), question.to(device), labels.to(device)
            outputs = model(image, question)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            _ , predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss = sum(losses) /len(losses)
    acc = correct / total

    return loss, acc
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        writer,
        epochs
    ):
    train_losses = []
    val_losses = []
    running_loss = 0.0
    best_val_acc = 0.0
    for epoch in range(epochs):
        batch_train_losses = []
        model.train()
        for idx, (images, questions, labels) in enumerate(tqdm(train_loader)):
            # img_grid = torchvision.utils.make_grid(images)
            # writer.add_image('four_vqa_images', img_grid)
            # if idx == 0:
            #     writer.add_graph(model, images)
            #     writer.close()

            images = images.to(device)
            questions = questions.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
            running_loss += loss.item()
            if idx % 2 == 0:
                writer.add_scalar(
                    'average training loss',
                    running_loss / (epoch * len(train_loader) + idx + 1),
                    epoch * len(train_loader) + idx
                    )
                writer.add_scalar(
                    'training loss',
                    loss.item(),
                    epoch * len(train_loader) + idx
                )
            if (epoch + 1) % 3 == 0:
                # Save model every 3 epochs
                save_model(model, f'model_runs/vqa_lstm_cnn_experiment_1_model_epoch_{epoch + 1}.pth')

        train_loss = sum(batch_train_losses)/len(batch_train_losses)
        train_losses.append(train_loss)

        if (epoch % 1 == 0) :
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            # Save accuracy and loss after every evaluation
            writer.add_scalar('validation loss', val_loss, epoch)
            writer.add_scalar('validation accuracy', val_acc, epoch)
            print(f'EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}\tVal Acc: {val_acc}')
            if val_acc > best_val_acc:
                # Save the best model based on validation accuracy
                best_val_acc = val_acc
                save_model(model, 'model_runs/vqa_lstm_cnn_experiment_1_best_model.pth')
        scheduler.step()
    return train_losses, val_losses


###Train model
lr = 1e-2
epochs = 1
weight_decay = 1e-5
scheduler_step_size = epochs * 0.5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = lr,
    weight_decay= weight_decay
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=scheduler_step_size,
    gamma=0.1
)
# evaluate(model, val_loader, criterion, device)
train_losses, val_losses = fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    writer,
    epochs
    )

# Save the final model after training completes
save_model(model, 'model_runs/vqa_lstm_cnn_experiment_1_final_model.pth')