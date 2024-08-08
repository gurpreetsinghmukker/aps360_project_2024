import os
from pathlib import Path
from pyexpat import model
from sympy import plot
import torch
import torchaudio
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


def gtzan_audio_tensors(dataset_path):
    # Collect the waveforms into a dictionary of genre index to list of waveforms
    genre_folders = sorted(list(dataset_path.glob('[!.]*')))
    genres = [folder.name.split("/")[-1] for folder in genre_folders]
    genre_index_map = {genre: i for i, genre in enumerate(genres)}
    audio_tensors = {
        genre_index_map[genre]: [] for genre in genres
    }

    references_sample_rate = None
    for genre_folder in genre_folders:
        files = sorted(list(genre_folder.glob('[!.]*.wav')))
        # if platform == 'nt':
        #     genre = genre_index_map[genre_folder.name.split("\\")[-1]]
        # else:
        genre = genre_index_map[genre_folder.name.split("/")[-1]]

        print(f'Processing Genre {genre} with {len(files)} files')
        for file in files:
            try:
                waveform, sample_rate = torchaudio.load(file)
                if references_sample_rate is None:
                    print(f'Setting references_sample_rate to {sample_rate}')
                    references_sample_rate = sample_rate
                if sample_rate != references_sample_rate:
                    print(f'Resampling {file} from {sample_rate} to {references_sample_rate}')
                    waveform = torchaudio.functional.resample(waveform, sample_rate, references_sample_rate)
                audio_tensors[genre].append(waveform)
            except RuntimeError:
                print(f'Error loading {file}')

    # For each genre, select p_test of the waveforms to be in the test set
    np.random.seed(SEED)
    audio_tensors_test = {}
    for genre, tensors in audio_tensors.items():
        n = len(tensors)
        n_test = int(n * p_test)
        test_indices = np.random.choice(n, n_test, replace=False)
        audio_tensors_test[genre] = [tensors[i] for i in range(n) if i in test_indices]
        audio_tensors[genre] = [tensors[i] for i in range(n) if i not in test_indices]
    
    return audio_tensors, audio_tensors_test, genres, references_sample_rate

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

def generate_gtzan_loaders(audio_tensors, genres, references_sample_rate, batch_size, mask_prob=0.8):
    # Return to the original dataset to train the classifier
    fold_generator = gtzan_generate_folds(audio_tensors, genres, references_sample_rate, mask_prob= mask_prob)
    train_dataset, val_dataset = next(fold_generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_classifier(embeddings_model_with_weights, classification_model, num_classes,  train_loader, val_loader, epochs,embeddings_learning_rate, classifier_learning_rate, embeddings_batch_size,  classifier_batch_size, output_dir: Path):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    contrastive_classifier = classification_model(embeddings_model_with_weights, num_classes)
    contrastive_classifier = contrastive_classifier.to(device)
    
    train_loss, val_loss, val_acc =  train_base_gtzan_classifier(contrastive_classifier, train_loader, val_loader, epochs=epochs, learning_rate=classifier_learning_rate, output_dir=output_dir)

    return train_loss, val_loss, val_acc, embedder_model_with_weights.size+"_"+str(embeddings_learning_rate)+"_"+str(embeddings_batch_size)+"_"+contrastive_classifier.name+"_"+str(classifier_learning_rate)+"_"+str(classifier_batch_size)

def plot_classifier_train_losses(t_losses, model_names, plot_save_path):
    # Plot the training losses
    contrastive_classifier_loss_fig, contrastive_classifier_loss_axs = plt.subplots(1, 1, figsize=(10, 5))
    for i in range(len(t_losses)):
        contrastive_classifier_loss_axs.plot(*zip(*t_losses[i]), label=f'{model_names[i]}')
    contrastive_classifier_loss_axs.legend()
    contrastive_classifier_loss_axs.set_xlabel('Step')
    contrastive_classifier_loss_axs.set_ylabel('Cross Entropy Loss')
    contrastive_classifier_loss_axs.set_title(f'Training Losses')
    
    plt.savefig(plot_save_path / 'train_losses.png')
    plt.show()

def plot_classifier_val_losses(v_losses, model_names, plot_save_path):
    # Plot the validation losses
    contrastive_classifier_loss_fig, contrastive_classifier_loss_axs = plt.subplots(1, 1, figsize=(10, 5))
    for i in range(len(v_losses)):
        contrastive_classifier_loss_axs.plot(*zip(*v_losses[i]), label=f'{model_names[i]}')
    contrastive_classifier_loss_axs.legend()
    contrastive_classifier_loss_axs.set_xlabel('Step')
    contrastive_classifier_loss_axs.set_ylabel('Cross Entropy Loss')
    contrastive_classifier_loss_axs.set_title(f'Validation Losses')
    
    plt.savefig(plot_save_path / 'val_losses.png')
    plt.show()
    
def plot_classifier_accuracy(v_accs, model_names, plot_save_path):
    contrastive_classifier_acc_fig, contrastive_classifier_acc_axs = plt.subplots(1, 1, figsize=(10, 5))

    for i in range(len(v_accs)):
        contrastive_classifier_acc_axs.plot(*zip(*v_accs[i]), label=f"{model_names[i]}")
        
    contrastive_classifier_acc_axs.legend()
    contrastive_classifier_acc_axs.set_xlabel('Step')
    contrastive_classifier_acc_axs.set_ylabel('Accuracy')
    contrastive_classifier_acc_axs.set_title(f'Classification Accuracies')
    
    plt.savefig(plot_save_path / 'val_accuracies.png')
    plt.show()

if __name__ == '__main__':
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    contrastive_output_dir = output_dir / 'contrastive'
    contrastive_output_dir.mkdir(exist_ok=True)
    
    plot_output_dir = output_dir / 'plots'
    plot_output_dir.mkdir(exist_ok=True)
    
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
    gtzan_path = Path("./genres/genres")
    audio_tensors, audio_tensor_test,  genres, reference_sample_rate = gtzan_audio_tensors(gtzan_path)
    # Parameters
    # gtzan_train_loader, gtzan_val_loader = generate_gtzan_loaders(audio_tensors, genres, reference_sample_rate, batch_size=32, mask_prob=0.8)
    
    
    embeddings_models = [GTZANContrastiveModelLarge, GTZANContrastiveModelXLarge]
    classification_models = [ContrastiveClassificationModel, ContrastiveClassificationModel_2, ContrastiveClassificationModel_3]
    classifier_learning_rates = [0.001, 0.0001]
    embeddings_batch_sizes = [256, 512]
    epochs = 100
    
    t_losses, v_losses, v_accs, model_names = [], [], [], []
    for embeddings_model in embeddings_models:
        for batch_size in embeddings_batch_sizes:
            for lr in classifier_learning_rates:
                for classification_model in classification_models:
                    #  Load the best contrastive embedding model
                    embedder_model_with_weights = embeddings_model(128)
                    model_name = get_model_name(embedder_model_with_weights.name, 0.001, batch_size)
                    
                    if (contrastive_output_dir / f'{model_name}_best.pth').exists():    
                        embedder_model_with_weights.load_state_dict(torch.load(contrastive_output_dir / f'{model_name}_best.pth'))
                        print(f"Training {model_name}")
                        gtzan_train_loader, gtzan_val_loader = generate_gtzan_loaders(audio_tensors, genres, reference_sample_rate, batch_size=32, mask_prob=0.8)
                        t_loss, v_loss, v_acc, model_name = train_classifier(embedder_model_with_weights, classification_model, len(audio_tensors),  gtzan_train_loader, gtzan_val_loader, epochs, 0.001, lr, batch_size, 32, output_dir)
                        t_losses.append(t_loss)
                        v_losses.append(v_loss)
                        v_accs.append(v_acc)
                        model_names.append(model_name)
                    else:
                        print(f"{model_name} does not exist")
                        
    plot_classifier_train_losses(t_losses, model_names, plot_output_dir)
    plot_classifier_val_losses(v_losses, model_names, plot_output_dir)
    plot_classifier_accuracy(v_accs, model_names, plot_output_dir)
    