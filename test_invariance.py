import torch
from dataset_classes.DEAM_CQT_shift_validation import DEAM_CQT_Shift_Validation_Set
from models.LSTM import LSTM_model
from models.GInv_LSTM import GInvariantLSTM_Model
from models.RNN import RNN_model
from models.MLP import MLP_model
from models.GInv_MLP import GInv_MLP_model
from models.GInv_RNN import GInvariantRNN_Model
import numpy as np

NUM_TO_ROLL_EACH_DIR = 10

def evaluate(model, test_loader, device):
    model.eval()
    avg_loss = 0.0
    num_vals = float(len(test_loader))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            data = torch.squeeze(data)
            output = torch.squeeze(model(data)) 
            sum_diffs = torch.zeros_like(output, dtype=output.dtype, device=output.device)     
            for i in range(NUM_TO_ROLL_EACH_DIR):
                rolled_data = torch.roll(output, i, dims=(0))
                diff = output - rolled_data
                diff = torch.pow(diff, 2)
                sum_diffs += diff
                rolled_data = torch.roll(output, -i, dims=(0))
                diff = output - rolled_data
                diff = torch.pow(diff, 2)
                sum_diffs += diff
            avg_loss += torch.norm(sum_diffs).item() / (NUM_TO_ROLL_EACH_DIR * 2)
    return avg_loss

def evaluate_invariance(model_str: str, dataset_str: str, test_loader):
    model_path = model_str + "_" + dataset_str + "_model.pt"

    model = torch.load(model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    avg_loss = evaluate(model, test_loader, device)
    print(f'Average invariance: {avg_loss:.4f}')

def main():
    annot_path = "deam_dataset/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv"
    audio_path = "deam_dataset/DEAM_audio/MEMD_audio/"
    TRANSFORM_PATH = "transforms/"
    TRANSFORM_NAME = "cqt_short_length"
    
    test_dataset = DEAM_CQT_Shift_Validation_Set(annot_path=annot_path, audio_path=audio_path, save_files=True, transform_path=TRANSFORM_PATH, transform_name=TRANSFORM_NAME, train=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    models = ['MLP', 'RNN', 'LSTM', 'GInv_MLP', 'GInv_RNN']
    datasets = ['sliding', 'circshift_sliding_restricted']
    for model_str in models:
        for dataset_str in datasets:
            print("Evaluating invariance for model="+model_str+" on dataset="+dataset_str)
            try:
                evaluate_invariance(model_str, dataset_str, test_loader)
            except Exception as err:
                print(err)

if __name__ == "__main__":
    main()