import os
import random

import matplotlib.pyplot as plt

from algorithms import bubble_sort_steps


def save_bubble_frames(n: int = 60, out_dir: str = "frames"):
    os.makedirs(out_dir, exist_ok=True)
    arr = [random.randint(1, 100) for _ in range(n)]

    gen = bubble_sort_steps(arr)

    frame = 0
    for state, (i, j) in gen:
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(state)), state)
        plt.title(f"Bubble Sort Frame {frame}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"frame_{frame:04d}.png"))
        plt.close()
        frame += 1

    print(f"Saved {frame} frames to {out_dir}/")


if __name__ == "__main__":
    save_bubble_frames()
