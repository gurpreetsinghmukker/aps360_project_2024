
import os
from pathlib import Path
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import dataloader
import numpy as np
import pickle
from transform_utilities import *
from visualization_utilities import *
from gtzan_dataset import *
from models import *
from mtg_contrastive import MTGContrastiveDataset
from mtg_contrastive_mel import MTG_Mel_ContrastiveDataset, worker_init_fn
from infonce_loss import InfoNCE
from barlow_twin_loss import BarlowTwinsLoss

def train_base_gtzan_classifier(model, train_loader, val_loader, epochs, learning_rate, output_dir: Path):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=epochs)

    t_loss_history = []
    v_loss_history = []
    v_acc_history = []

    step = 0
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()

        for (waveform, mel_spectrogram, label) in tqdm(train_loader):
            mel_spectrogram = mel_spectrogram.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(mel_spectrogram)
            loss = criterion(output, label)
            t_loss_history.append((step, loss.item()))
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

        model.eval()

        with torch.no_grad():
            val_losses = []
            val_accuracies = []
            for (waveform, mel_spectrogram, label) in val_loader:
                mel_spectrogram = mel_spectrogram.to(device)
                label = label.to(device)

                output = model(mel_spectrogram)
                val_loss = criterion(output, label)
                val_losses.append(val_loss.item())
                val_acc = (output.argmax(dim=1) == label).float().mean()
                val_accuracies.append(val_acc.item())

            val_loss = torch.tensor(val_losses).mean()
            v_loss_history.append((step, val_loss.item()))
            val_acc = torch.tensor(val_accuracies).mean()
            v_acc_history.append((step, val_acc.item()))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / 'best.pth')

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.4f}")

    return t_loss_history, v_loss_history, v_acc_history


def get_model_name(model_name, lr, batch_size):
    return f"{model_name}_LR({lr})_BS({batch_size})"

def train_contrastive_model(model, train_loader, val_loader, epochs, lr, batch_size, criterion, output_dir: Path, t_loss_history=None, v_loss_history=None, checkpoint_file=None, device = None):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    criterion = criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    t_loss_history = [] if t_loss_history is None else t_loss_history
    v_loss_history = [] if v_loss_history is None else v_loss_history

    step = 0
    best_val_loss = np.inf

    if checkpoint_file is not None:
        print(f"Recovering model from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]
        recovered_epoch = checkpoint["epoch"]
        recovered_t_loss_history = checkpoint["t_loss_history"]
        recovered_v_loss_history = checkpoint["v_loss_history"]

        print(f"Recovered model from epoch {recovered_epoch} with best val loss {best_val_loss:.4f}")

        # Copy the recovered loss histories into the current ones
        t_loss_history.extend(recovered_t_loss_history)
        v_loss_history.extend(recovered_v_loss_history)
    else:
        recovered_epoch = 0

    for epoch in range(recovered_epoch, epochs):
        model.train()

        train_losses = []
        for (waveform1, waveform2), (mel_spectrogram1, mel_spectrogram2), label in tqdm(train_loader):
            
            loss_acc = 0
            optimizer.zero_grad()
            if batch_size > 64:
                for iter in range((mel_spectrogram1.shape[0] // 64)):
                    mini_batch_1 = mel_spectrogram1[iter*64:(iter+1)*64].to(device)
                    mini_batch_2 = mel_spectrogram2[iter*64:(iter+1)*64].to(device)
                    mini_batch = [mini_batch_1, mini_batch_2]
                    output = model(mini_batch)
                    loss = criterion(output)
                    loss_acc += loss.item()
                    loss.backward()
                if mel_spectrogram1.shape[0] % 64 != 0:
                    mini_batch_1 = mel_spectrogram1[(iter+1)*64:].to(device)
                    mini_batch_2 = mel_spectrogram2[(iter+1)*64:].to(device)
                    mini_batch = [mini_batch_1, mini_batch_2]
                    output = model(mini_batch)
                    loss = criterion(output)
                    loss_acc += loss.item()
                    loss.backward()
            else:
                mel_spectrogram1 = mel_spectrogram1.to(device)
                mel_spectrogram2 = mel_spectrogram2.to(device)
                mel_spectrogram = [mel_spectrogram1, mel_spectrogram2]
                output = model(mel_spectrogram)
                loss = criterion(output)
                loss.backward()
            # mel_spectrogram1 = mel_spectrogram1.to(device)
            # mel_spectrogram2 = mel_spectrogram2.to(device)
            # mel_spectrogram = [mel_spectrogram1, mel_spectrogram2]
            # Stack the spectrograms along the batch dimension
            # mel_spectrogram = torch.cat([mel_spectrogram1, mel_spectrogram2], dim=0)

            # print(mel_spectrogram.shape)

            # output = model(mel_spectrogram)

            # Split the output into the original batches
            # output1, output2 = torch.split(output, output.shape[0] // 2, dim=0)

            # loss = criterion(output)
            t_loss_history.append((step, loss_acc))
            train_losses.append(loss_acc)
            
            optimizer.step()
            scheduler.step()
            step += 1
        avg_train_loss = torch.tensor(train_losses).mean()

        model.eval()
        # print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss.item():.4f}")
        with torch.no_grad():

            val_losses = []
            val_accuracies = []
            for (waveform1, waveform2), (mel_spectrogram1, mel_spectrogram2), label in tqdm(val_loader):
                if batch_size > 64:
                    for iter in range((mel_spectrogram1.shape[0] // 64)):
                        mini_batch_1 = mel_spectrogram1[iter*64:(iter+1)*64].to(device)
                        mini_batch_2 = mel_spectrogram2[iter*64:(iter+1)*64].to(device)
                        mini_batch = [mini_batch_1, mini_batch_2]
                        output = model(mini_batch)
                        val_loss = criterion(output)
                        val_losses.append(val_loss.item())
                        
                    if mel_spectrogram1.shape[0] % 64 != 0:
                        mini_batch_1 = mel_spectrogram1[(iter+1)*64:].to(device)
                        mini_batch_2 = mel_spectrogram2[(iter+1)*64:].to(device)
                        mini_batch = [mini_batch_1, mini_batch_2]
                        output = model(mini_batch)
                        val_loss = criterion(output)
                        val_losses.append(val_loss.item())
                else:
                    mel_spectrogram1 = mel_spectrogram1.to(device)
                    mel_spectrogram2 = mel_spectrogram2.to(device)
                    mel_spectrogram = [mel_spectrogram1, mel_spectrogram2]
                    output = model(mel_spectrogram)
                    val_loss = criterion(output)
                    val_losses.append(val_loss.item())
                

            val_loss = torch.tensor(val_losses).mean()
            v_loss_history.append((step, val_loss.item()))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / f"{get_model_name(model.name, lr, batch_size)}_best.pth")

            # Save the latest model
            save_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "t_loss_history": t_loss_history,
                "v_loss_history": v_loss_history
            }
            torch.save(save_data, output_dir / f"{get_model_name(model.name, lr, batch_size)}_latest_checkpoint.pth")
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    return t_loss_history, v_loss_history

            


def train_model_with_params(batch_size, learning_rate, epochs, criterion, model, output_dir):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    batch_size = batch_size
    learning_rate = learning_rate

    # Using the MTG Dataset
    train_folders = [
        "00", "01", "02", "03",
        "04", "05", "06", "07",
        "08", "09", "10", "11",
        "12", "13", "14", "15",
        "16", "17", "18", "19", 
        "21", "22", "23", "24",
        "25", "26", "27", "28",
        "29", "30"
    ]
    
    # train_folders = ["00"]
    
    val_folders = [
        "20"
    ]

    mtg_train_dataset = MTG_Mel_ContrastiveDataset(mtg_mel_path, references_sample_rate, mask_prob=0.8, samples_per_file=15, folder_whitelist=train_folders, concurrent_files=batch_size)
    mtg_val_dataset = MTG_Mel_ContrastiveDataset(mtg_mel_path, references_sample_rate, mask_prob=0.8, samples_per_file=15, folder_whitelist=val_folders, max_files=2*batch_size, concurrent_files=batch_size)

    train_loader = DataLoader(mtg_train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, prefetch_factor=2)
    val_loader = DataLoader(mtg_val_dataset, batch_size=batch_size, shuffle=False)

    contrastive_embedder_model = model
    contrastive_embedder_model = contrastive_embedder_model.to(device)


    USE_LATEST_CHECKPOINT = True
    checkpoint_file = output_dir / f'{get_model_name(contrastive_embedder_model.name, learning_rate, batch_size)}_latest_checkpoint.pth' if USE_LATEST_CHECKPOINT else None
    if checkpoint_file is not None and not checkpoint_file.exists():
        print(f"Checkpoint file {checkpoint_file} not found, starting with a fresh model")
        checkpoint_file = None
    else:
        print(f"Recovering model from {checkpoint_file}")


    t_loss, v_loss = [], []
    _ = train_contrastive_model(
        contrastive_embedder_model,
        train_loader, val_loader, epochs=epochs, lr=learning_rate, batch_size=batch_size, criterion = criterion,
        t_loss_history=t_loss, v_loss_history=v_loss,
        output_dir=output_dir,
        checkpoint_file=checkpoint_file,
        device = device
    )
    
    losses = {'train_loss': t_loss, 'val_loss': v_loss}
    with open(output_dir / f'{get_model_name(contrastive_embedder_model.name, learning_rate, batch_size)}.pkl', 'wb') as file:
        pickle.dump(losses, file)
    return




def plot_losses(t_loss, v_loss):
    # Plot the losses
    contrastive_loss_fig, contrastive_loss_axs = plt.subplots(1, 1, figsize=(10, 5))
    contrastive_loss_axs.plot(*zip(*t_loss), label='Training Loss')
    contrastive_loss_axs.plot(*zip(*v_loss), label='Validation Loss')
    # contrastive_loss_axs.plot(*zip(*t_loss), label='Training Loss')
    contrastive_loss_axs.legend()
    contrastive_loss_axs.set_xlabel('Step')
    contrastive_loss_axs.set_ylabel('InfoNCE Loss')
    plt.show()


if __name__ == '__main__':
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    p_test = 0.2
    p_train_val = 1 - p_test

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)


    # Path for the deep model trained end to end on GTZAN
    deep_base_output_dir = output_dir / 'base_gtzan'
    deep_base_output_dir.mkdir(exist_ok=True)

    # Define the path for the checkpoints for the pretrained contrastive model
    contrastive_output_dir = output_dir / 'contrastive'
    contrastive_output_dir.mkdir(exist_ok=True)

    # Define the path for the checkpoints for the pretrained barlow model
    barlow_contrastive_output_dir = output_dir / 'barlow_contrastive'
    barlow_contrastive_output_dir.mkdir(exist_ok=True)

    # Path for the mlp that accepts the embeddings (un-normalized) from the contrastive model
    contrastive_classifier_output_dir = output_dir / 'contrastive_classifier'
    contrastive_classifier_output_dir.mkdir(exist_ok=True)

    # Path for the mlp that accepts the logits from the contrastive model
    contrastive_classifier_embedder_only_output_dir = output_dir / 'contrastive_classifier_embedder_only'
    contrastive_classifier_embedder_only_output_dir.mkdir(exist_ok=True)

    # Path for the mlp that accepts the embeddings (normalized) from the barlow model
    contrastive_classifier_barlow_output_dir = output_dir / 'contrastive_classifier_barlow'
    contrastive_classifier_barlow_output_dir.mkdir(exist_ok=True)

    # Path for the mlp that accepts the logits from the barlow model
    contrastive_classifier_barlow_embedder_only_output_dir = output_dir / 'contrastive_classifier_barlow_embedder_only'
    contrastive_classifier_barlow_embedder_only_output_dir.mkdir(exist_ok=True)

    # Path for baseline random forest
    baseline_random_forest_output_dir = output_dir / 'baseline_random_forest'
    baseline_random_forest_output_dir.mkdir(exist_ok=True)

    # Path for the random forest that uses contrastive embeddings
    contrastive_random_forest_output_dir = output_dir / 'contrastive_random_forest'
    contrastive_random_forest_output_dir.mkdir(exist_ok=True)


    references_sample_rate = 22050
    mtg_mel_path = mtg_path = Path('') / 'mtg_mel'
    mtg_dataset = MTG_Mel_ContrastiveDataset(mtg_mel_path, references_sample_rate, mask_prob=0)
    print(f"Dataset size: {len(mtg_dataset)}")
    
    model = GTZANContrastiveModelLarge(128)
    criterion = InfoNCE()
    output_dir = contrastive_output_dir
    batch_size = 256
    learning_rate = 0.001
    epochs = 25
    train_model_with_params(batch_size=batch_size, learning_rate=learning_rate, epochs = epochs, model=model, output_dir = output_dir, criterion = criterion )
