import os
import numpy as np
import matplotlib.pyplot as plt


def _smooth(data: list, window: int) -> np.ndarray:
    """Rolling mean; falls back to raw values when shorter than the window."""
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def _x_smooth(n: int, window: int) -> np.ndarray:
    """Iteration indices aligned to the smoothed array length."""
    return np.arange(window - 1, n) if n >= window else np.arange(n)


def plot_rewards(history: dict, algo_type: str = "MARL",
                 window: int = 20, save_path: str = None) -> None:
    """
    Plot the expected return (mean reward) for attacker and defender over training iterations.

    Args:
        history:   dict with keys 'attacker_reward' and 'defender_reward' (list[float])
        algo_type: algorithm label shown in the title (e.g. 'IPPO', 'MAPPO')
        window:    rolling-mean smoothing window in iterations
        save_path: if given, save PNG to this path; otherwise display interactively
    """
    att  = history.get('attacker_reward', [])
    def_ = history.get('defender_reward', [])

    fig, ax = plt.subplots(figsize=(12, 5))

    if att:
        xs = _x_smooth(len(att), window)
        ax.plot(att,  alpha=0.2, color='tomato')
        ax.plot(xs, _smooth(att, window),  color='tomato',    linewidth=2,
                label=f'Attacker (smooth w={window})')
    if def_:
        xs = _x_smooth(len(def_), window)
        ax.plot(def_, alpha=0.2, color='steelblue')
        ax.plot(xs, _smooth(def_, window), color='steelblue', linewidth=2,
                label=f'Defender (smooth w={window})')

    ax.set_title(f'{algo_type.upper()} — Expected Return per Agent')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Mean Episode Reward')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_steps(history: dict, algo_type: str = "MARL",
               window: int = 20, save_path: str = None) -> None:
    """
    Plot the mean number of steps per episode over training iterations.

    A decreasing trend means the attacker finds the goal faster; an increasing
    trend means the defender is forcing longer (costlier) attacks.

    Args:
        history:   dict with key 'episode_len_mean' (list[float])
        algo_type: algorithm label shown in the title
        window:    rolling-mean smoothing window in iterations
        save_path: if given, save PNG to this path; otherwise display interactively
    """
    data = history.get('episode_len_mean', [])
    if not data:
        print("[plot] 'episode_len_mean' not present in history — skipping.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    xs = _x_smooth(len(data), window)
    ax.plot(data, alpha=0.2, color='mediumseagreen')
    ax.plot(xs, _smooth(data, window), color='mediumseagreen', linewidth=2,
            label=f'Avg steps (smooth w={window})')

    ax.set_title(f'{algo_type.upper()} — Mean Steps per Episode')
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Steps per Episode')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path: str) -> None:
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[plot] Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
