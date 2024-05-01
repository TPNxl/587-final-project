import torch
from absl import app, flags
from dataset_classes.DEAM_CQT_circshift_and_sliding import DEAM_CQT_Dataset_With_CircShift_Sliding
from dataset_classes.DEAM_CQT_circshift_and_sliding_restricted import DEAM_CQT_Dataset_With_CircShift_Sliding_Restricted
from dataset_classes.DEAM_CQT_sliding_efficient import DEAM_CQT_Dataset_Sliding_Efficient
# from dataset_classes.DEAM_CQT_extended_len import DEAM_CQT_Dataset_Sliding
from models.LSTM import LSTM_model
from models.GInv_LSTM import GInvariantLSTM_Model
from models.RNN import RNN_model
from models.MLP import MLP_model
from models.GInv_MLP import GInv_MLP_model
from models.GInv_RNN import GInvariantRNN_Model
import copy
import librosa
import numpy as np
import matplotlib.pyplot as plt
import wandb

N_EPOCHS = 100
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HIDDEN_SIZE = 30
NUM_LAYERS = 3
ANNOT_PATH = "deam_dataset/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv"
AUDIO_PATH = "deam_dataset/DEAM_audio/MEMD_audio/"
TRANSFORM_PATH = "transforms/"
TRANSFORM_NAME = "cqt_short_length"
LEARNING_RATE = 1e-5    # decrease
DROPOUT = 0.2
MAX_NONDEC_EPOCHS = 50

def chroma_cqt(y, sr, hop_length):
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

def cqt(y, sr, hop_length):
    return np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length))

TRANSFORM_FUNC = cqt
DATASET_CLASS = DEAM_CQT_Dataset_Sliding_Efficient
# DATASET_NAME = "circshift_sliding"

flags.DEFINE_enum('dataset_name', 'sliding', ['sliding', 'circshift_sliding','circshift_sliding_restricted'], 'Dataset type')
flags.DEFINE_enum('model_type', 'LSTM', ['MLP', 'RNN', 'LSTM', 'GInv_MLP', 'GInv_RNN', 'GInv_LSTM'], 'Dataset type')

FLAGS = flags.FLAGS

def evaluate(model, test_loader, criterion, device):
    model.eval()
    avg_loss = 0.0
    num_vals = float(len(test_loader))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = torch.squeeze(model(data))        
            avg_loss += criterion(output, target) / num_vals
    return avg_loss

def train(model,
         *,
         train_loader,
         test_loader,
         optimizer,
         criterion,
         scheduler,
         num_epochs,
         device):
    wandb.watch(model, criterion, log='all', log_freq=10)
    best_model = None
    best_acc = 100.0
    nondecreasing_acc = 0
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        model.train()
        n = 0
        loss_sum = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx % 5000 == 0:
               print("Batch ", batch_idx)
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = torch.squeeze(model(data))        
            loss = criterion(output, target)
            with torch.no_grad():
                loss_sum += loss
                n += 1
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
          loss_sum /= n
          scheduler.step()
          accuracy = evaluate(model, test_loader, criterion, device)
          print(f'Epoch {epoch + 1}/{num_epochs}, Train loss: {loss_sum:.4f}, Test loss: {accuracy:.4f}')
          wandb.log({"epoch": epoch, "loss": loss_sum}, step=epoch)
          train_loss.append(loss_sum.to("cpu").item())
          test_loss.append(accuracy.to("cpu"))
        

        # if best_model is None or accuracy < best_acc:
        #     best_acc = accuracy
        #     best_model = copy.deepcopy(model)
        #     nondecreasing_acc = 0
        # else:
        #     nondecreasing_acc += 1 

        # if nondecreasing_acc > MAX_NONDEC_EPOCHS:
        #     break

        if accuracy < best_acc:
             best_acc = accuracy
             nondecreasing_acc = 0
        else:
             nondecreasing_acc += 1 

        if nondecreasing_acc > MAX_NONDEC_EPOCHS:
             break
        

    print('Training finished!')

    return model, train_loss, test_loss


def main(argv):
    if FLAGS.dataset_name == "sliding":
        DATASET_CLASS = DEAM_CQT_Dataset_Sliding_Efficient
        BATCH_SIZE = 4
    elif FLAGS.dataset_name == "circshift_sliding":
        DATASET_CLASS = DEAM_CQT_Dataset_With_CircShift_Sliding
        BATCH_SIZE = 100
    elif FLAGS.dataset_name == "circshift_sliding_restricted":
        DATASET_CLASS = DEAM_CQT_Dataset_With_CircShift_Sliding_Restricted
        BATCH_SIZE = 40

    # torch.autograd.set_detect_anomaly(True)
    train_dataset = DATASET_CLASS(annot_path=ANNOT_PATH, audio_path=AUDIO_PATH, save_files=True, transform_path=TRANSFORM_PATH, transform_name=TRANSFORM_NAME, transform_func=TRANSFORM_FUNC, train=True)
    test_dataset = DATASET_CLASS(annot_path=ANNOT_PATH, audio_path=AUDIO_PATH, save_files=True, transform_path=TRANSFORM_PATH, transform_name=TRANSFORM_NAME, transform_func=TRANSFORM_FUNC, train=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=False)
    
    input_size = train_dataset.__getitem__(0)[0].shape[1]
    # model = LSTM_model(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, out_size=1, dropout=DROPOUT)

    if FLAGS.model_type == "MLP":
      model = MLP_model(input_size=input_size, hidden_size=HIDDEN_SIZE, out_size=1, num_layers=1)
      LEARNING_RATE = 1e-5
    if FLAGS.model_type == "RNN":
      model = RNN_model(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, out_size=1, dropout=DROPOUT)
      LEARNING_RATE = 1e-5
    if FLAGS.model_type == "LSTM":
      model = LSTM_model(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, out_size=1, dropout=DROPOUT)
      LEARNING_RATE = 1e-5
    if FLAGS.model_type == "GInv_MLP":
      model = GInv_MLP_model(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=1, out_size=1)
      LEARNING_RATE = 3e-5
    if FLAGS.model_type == "GInv_RNN":
      model = GInvariantRNN_Model(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS-1, out_size=1, dropout=DROPOUT)
      LEARNING_RATE = 3e-5
    if FLAGS.model_type == "GInv_LSTM":
      model = GInvariantLSTM_Model(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS-1, out_size=1, dropout=DROPOUT)
      LEARNING_RATE = 3e-5

    config = dict(
        model=FLAGS.model_type,
        batch_size=BATCH_SIZE,
        dataset=FLAGS.dataset_name,
        learning_rate=LEARNING_RATE,
    )

    wandb.init(project="pytorch-demo", config=config)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
    #     torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.pow(0.9, 0.2)),
    #     torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01),
    #     torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # ], milestones=[10, 11])
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) # increase gamma
    # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.power(0.9, 0.1))
    model = model.to(DEVICE)

    print("Done with init on " + FLAGS.model_type + " training on " + FLAGS.dataset_name)

    best_model, train_loss, test_loss = train(model,
          train_loader=train_loader,
          test_loader=test_loader,
          criterion=criterion,
          optimizer=optimizer,

          num_epochs=N_EPOCHS,
          device=DEVICE,
          scheduler=scheduler)
    
    torch.save(best_model, FLAGS.model_type +"_"+FLAGS.dataset_name+"_model.pt")

    vals = np.arange(1, len(train_loss)+1)
    plt.plot(vals, train_loss, 'r')
    plt.plot(vals, test_loss, 'g')
    plt.title("Training with model="+FLAGS.model_type+" and dataset="+FLAGS.dataset_name)
    plt.xlabel("Epoch #")
    plt.savefig(FLAGS.model_type+"_"+FLAGS.dataset_name+".png")

    # print("Evaluating on test set")
    # accuracy = evaluate(best_model, test_loader, criterion, DEVICE)
    # print(f'Final test set loss: {accuracy:.4f}')

    # (chroma, annots) = test_dataset.__getitem__(0)
    # print(model(chroma))
    # print(annots)

if __name__ == '__main__':
    app.run(main)
