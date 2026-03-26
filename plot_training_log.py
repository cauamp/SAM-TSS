import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_log(run_path):
    """Parse and plot the automated_log.txt file from a run folder."""

    run_path = Path(run_path)
    log_file = run_path / "automated_log.txt"

    if not log_file.exists():
        raise FileNotFoundError(f"automated_log.txt not found at {log_file}")

    # Read the log file with flexible whitespace handling
    df = pd.read_csv(log_file, sep=r"\s+", engine="python")

    # Ensure we have the expected columns
    expected_cols = ["Epoch", "Train-loss", "Val-loss", "Train-mIoU", "Val-mIoU", "learningRate"]
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Missing columns in log file. Expected: {expected_cols}, Got: {list(df.columns)}")

    # Convert columns to numeric, coercing errors to NaN
    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with NaN values in critical columns
    df = df.dropna(subset=["Epoch", "Val-loss", "Val-mIoU", "learningRate"])

    if len(df) == 0:
        raise ValueError("No valid data found in log file after parsing")

    # Extract run name from path
    run_name = run_path.name

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"Training Log: {run_name}", fontsize=16, fontweight="bold")

    # Plot 1: Loss
    ax1 = axes[0]
    ax1.plot(df["Epoch"], df["Train-loss"], label="Train Loss", marker="o", markersize=3, linewidth=1.5)
    ax1.plot(df["Epoch"], df["Val-loss"], label="Val Loss", marker="s", markersize=3, linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot 2: mIoU
    ax2 = axes[1]
    # Only plot Train mIoU if there are non-zero values
    if df["Train-mIoU"].notna().any() and (df["Train-mIoU"] != 0).any():
        ax2.plot(df["Epoch"], df["Train-mIoU"], label="Train mIoU", marker="o", markersize=3, linewidth=1.5)

    ax2.plot(df["Epoch"], df["Val-mIoU"], label="Val mIoU", marker="s", markersize=3, linewidth=1.5)

    # Find and mark best validation mIoU
    best_idx = df["Val-mIoU"].idxmax()
    best_epoch = df.loc[best_idx, "Epoch"]
    best_miou = df.loc[best_idx, "Val-mIoU"]
    ax2.plot(
        best_epoch,
        best_miou,
        marker="*",
        markersize=15,
        color="red",
        label=f"Best Val mIoU ({best_miou:.2f}% @ Epoch {int(best_epoch)})",
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU (%)")
    ax2.set_title("mIoU over Epochs")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Learning Rate
    ax3 = axes[2]
    ax3.plot(
        df["Epoch"], df["learningRate"], label="Learning Rate", marker="o", markersize=3, linewidth=1.5, color="green"
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.set_yscale("log")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    # Save the figure
    output_path = run_path / "training_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved to: {output_path}")

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"Training Summary: {run_name}")
    print(f"{'=' * 60}")
    print(f"Total Epochs: {len(df)}")
    print(f"\nBest Validation mIoU: {best_miou:.2f}% (Epoch {int(best_epoch)})")
    print(f"Final Train Loss: {df.iloc[-1]['Train-loss']:.4f}")
    print(f"Final Val Loss: {df.iloc[-1]['Val-loss']:.4f}")
    if (df["Train-mIoU"] != 0).any():
        print(f"Final Train mIoU: {df.iloc[-1]['Train-mIoU']:.2f}%")
    else:
        print("Final Train mIoU: Not tracked")
    print(f"Final Val mIoU: {df.iloc[-1]['Val-mIoU']:.2f}%")
    print(f"\nInitial Learning Rate: {df.iloc[0]['learningRate']:.6f}")
    print(f"Final Learning Rate: {df.iloc[-1]['learningRate']:.6f}")
    print(f"{'=' * 60}\n")

    plt.close()


if __name__ == "__main__":
    import glob

    for run_path in glob.glob("./runs/*/"):
        if "746" not in run_path:
            continue
        try:
            plot_training_log(run_path)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
