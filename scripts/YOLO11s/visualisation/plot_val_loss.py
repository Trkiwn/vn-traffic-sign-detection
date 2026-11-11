"""Plot the saved retrospective YOLO11s validation loss history (no training-loss curve).

Usage from repository root:
    python scripts/YOLO11s/visualisation/plot_val_loss.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=Path('results/YOLO11s/min_val_loss/val_loss_history.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/plots/yolo11s_validation_loss.png'))
    args = parser.parse_args()
    if not args.history.is_file():
        parser.error(f'Validation loss history not found: {args.history}')
    history = json.loads(args.history.read_text(encoding='utf-8'))
    if not isinstance(history, list) or not history:
        parser.error('Expected nonempty list of epoch records')
    epochs = [int(x['epoch']) for x in history]
    values = [float(x['total_val_loss']) for x in history]
    if len(epochs) != 50 or epochs != list(range(1, 51)):
        parser.error('Expected exactly 50 consecutive epochs (1 to 50)')
    min_at = min(range(len(values)), key=lambda i: (values[i], epochs[i]))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(epochs, values, linewidth=1.7, label='Validation loss (box + cls + DFL)')
    ax.scatter([epochs[min_at]], [values[min_at]], s=70, zorder=3,
               label=f'Minimum at epoch {epochs[min_at]}: {values[min_at]:.6f}')
    ax.axvline(epochs[min_at], linestyle='--', alpha=0.6)
    ax.set(xlabel='Epoch', ylabel='Mean per-batch validation loss',
           title='YOLO11s: retrospective validation loss over 50 saved checkpoints')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f'Saved: {args.output}; selected epoch: {epochs[min_at]}')


if __name__ == '__main__':
    main()
