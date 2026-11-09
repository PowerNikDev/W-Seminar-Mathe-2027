import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from helpers import to_cartesian
from recorder import Recorder
import argparse

def visualize(times, positions, n):
    frames = len(times) 
    cart = np.zeros((frames, n, 3))
    for f in range(frames):
        for i in range(n):
            cart[f, i] = to_cartesian(positions[f, i])

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    points = [ax.plot([], [], [], 'o', markersize=8)[0] for _ in range(n)]
    lines  = [ax.plot([], [], [], '-', alpha=0.3)[0]   for _ in range(n)]

    max_val = np.max(np.abs(cart)) * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def update(frame):
        for i in range(n):
            pos = cart[frame, i]
            points[i].set_data([pos[0]], [pos[1]])
            points[i].set_3d_properties([pos[2]])

            traj = cart[:frame+1, i]
            lines[i].set_data(traj[:, 0], traj[:, 1])
            lines[i].set_3d_properties(traj[:, 2])

        ax.set_title(f"t = {times[frame]:.3f}")
        return points + lines

    ani = animation.FuncAnimation(
        fig, update,
        frames=frames,
        interval=20,
        cache_frame_data=False
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualisiert eine gespeicherte N-Körper-Simulation."
    )
    parser.add_argument("path", type=str,
                        help="Pfad zur .npz Datei (mit oder ohne .npz)")
    args = parser.parse_args()

    path = args.path if args.path.endswith(".npz") else f"{args.path}.npz"

    times, positions, velocities, masses = Recorder.load(path)
    print(positions)
    n = positions.shape[1]  # Anzahl Körper aus Daten lesen

    print(f"Lade {path}: {len(times)} Frames, {n} Körper")
    visualize(times, positions, n)