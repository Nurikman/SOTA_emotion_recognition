import os
import pickle
import matplotlib.pyplot as plt

base_dir = "D:/Yeskendir_files/results2"
num_folds = 4

for fold in range(1, num_folds + 1):

    fold_dir = os.path.join(base_dir, f"fold {fold}")

    # Load data
    train_loss = pickle.load(open(os.path.join(fold_dir, "train_loss.pkl"), "rb"))
    val_loss   = pickle.load(open(os.path.join(fold_dir, "val_loss.pkl"),   "rb"))
    train_acc  = pickle.load(open(os.path.join(fold_dir, "train_acc.pkl"),  "rb"))
    val_acc    = pickle.load(open(os.path.join(fold_dir, "val_acc.pkl"),    "rb"))

    # Convert to lists
    epochs = list(train_loss.keys())
    train_loss_vals = list(train_loss.values())
    val_loss_vals   = list(val_loss.values())
    train_acc_vals  = list(train_acc.values())
    val_acc_vals    = list(val_acc.values())

    # -----------------------
    # 1) LOSS PLOT
    # -----------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_vals, label="Training Loss")
    plt.plot(epochs, val_loss_vals, label="Validation Loss")
    plt.title(f"Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    loss_path = os.path.join(fold_dir, f"fold_{fold}_loss_curve.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot → {loss_path}")

    # -----------------------
    # 2) ACCURACY PLOT
    # -----------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc_vals, label="Training Accuracy")
    plt.plot(epochs, val_acc_vals, label="Validation Accuracy")
    plt.title(f"Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    acc_path = os.path.join(fold_dir, f"fold_{fold}_accuracy_curve.png")
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy plot → {acc_path}")
