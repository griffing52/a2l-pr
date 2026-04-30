import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from a2l_pr.utils.trajectory import TrajectoryAnalyzer

def visualize_trajectory_comparison(
    original: Dict,
    perturbed: Dict,
    title: str = "Trajectory Comparison"
) -> None:
    """Plot original vs perturbed trajectory side by side.
    
    Args:
        original: original trajectory dict
        perturbed: perturbed trajectory dict
        title: plot title
    """
    analyzer_orig = TrajectoryAnalyzer(original)
    analyzer_pert = TrajectoryAnalyzer(perturbed)
    
    ee_orig = analyzer_orig.get_ee_position()
    ee_pert = analyzer_pert.get_ee_position()
    gripper_orig = analyzer_orig.get_gripper_state()
    gripper_pert = analyzer_pert.get_gripper_state()
    transitions_orig = analyzer_orig.find_gripper_transitions()
    transitions_pert = analyzer_pert.find_gripper_transitions()
    
    if len(ee_orig) == 0:
        print("No EE position found in trajectories")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=16)
    
    # 3D path visualization (XY plane)
    axes[0].plot(ee_orig[:, 0], ee_orig[:, 1], 'b-', alpha=0.7, label='Original')
    axes[0].plot(ee_pert[:, 0], ee_pert[:, 1], 'r--', alpha=0.7, label='Perturbed')
    axes[0].scatter(*ee_orig[0, :2], c='b', marker='o', s=100, label='Start')
    axes[0].scatter(*ee_orig[-1, :2], c='g', marker='s', s=100, label='End')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('XY Path')

    close_orig = [t for t, direction in transitions_orig if direction == 'close' and 0 <= t < len(ee_orig)]
    open_orig = [t for t, direction in transitions_orig if direction == 'open' and 0 <= t < len(ee_orig)]
    close_pert = [t for t, direction in transitions_pert if direction == 'close' and 0 <= t < len(ee_pert)]
    open_pert = [t for t, direction in transitions_pert if direction == 'open' and 0 <= t < len(ee_pert)]

    if close_orig:
        axes[0].scatter(ee_orig[close_orig, 0], ee_orig[close_orig, 1], c='navy', marker='v', s=70, label='Orig Close')
    if open_orig:
        axes[0].scatter(ee_orig[open_orig, 0], ee_orig[open_orig, 1], c='cyan', marker='^', s=70, label='Orig Open')
    if close_pert:
        axes[0].scatter(ee_pert[close_pert, 0], ee_pert[close_pert, 1], c='darkred', marker='v', s=70, label='Pert Close')
    if open_pert:
        axes[0].scatter(ee_pert[open_pert, 0], ee_pert[open_pert, 1], c='orange', marker='^', s=70, label='Pert Open')

    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Z trajectory
    t_orig = np.arange(len(ee_orig))
    t_pert = np.arange(len(ee_pert))
    axes[1].plot(t_orig, ee_orig[:, 2], 'b-', alpha=0.7, label='Original Z')
    axes[1].plot(t_pert, ee_pert[:, 2], 'r--', alpha=0.7, label='Perturbed Z')

    if close_orig:
        axes[1].scatter(close_orig, ee_orig[close_orig, 2], c='navy', marker='v', s=55, label='Orig Close (Z)')
    if open_orig:
        axes[1].scatter(open_orig, ee_orig[open_orig, 2], c='cyan', marker='^', s=55, label='Orig Open (Z)')
    if close_pert:
        axes[1].scatter(close_pert, ee_pert[close_pert, 2], c='darkred', marker='v', s=55, label='Pert Close (Z)')
    if open_pert:
        axes[1].scatter(open_pert, ee_pert[open_pert, 2], c='orange', marker='^', s=55, label='Pert Open (Z)')

    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Z')
    axes[1].set_title('Vertical Trajectory')
    axes[1].set_ylim(bottom=0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Gripper timeline
    axes[2].set_title('Gripper Signal')
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('Signal')

    if gripper_orig is not None and len(gripper_orig) > 0:
        gripper_orig_1d = TrajectoryAnalyzer._to_1d_signal(gripper_orig)
        t_gripper_orig = np.arange(len(gripper_orig_1d))
        axes[2].plot(t_gripper_orig, gripper_orig_1d, 'b-', alpha=0.75, label='Original Gripper')
        if close_orig:
            axes[2].scatter(close_orig, gripper_orig_1d[close_orig], c='navy', marker='v', s=55)
        if open_orig:
            axes[2].scatter(open_orig, gripper_orig_1d[open_orig], c='cyan', marker='^', s=55)

    if gripper_pert is not None and len(gripper_pert) > 0:
        gripper_pert_1d = TrajectoryAnalyzer._to_1d_signal(gripper_pert)
        t_gripper_pert = np.arange(len(gripper_pert_1d))
        axes[2].plot(t_gripper_pert, gripper_pert_1d, 'r--', alpha=0.75, label='Perturbed Gripper')
        if close_pert:
            axes[2].scatter(close_pert, gripper_pert_1d[close_pert], c='darkred', marker='v', s=55)
        if open_pert:
            axes[2].scatter(open_pert, gripper_pert_1d[open_pert], c='orange', marker='^', s=55)

    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_trajectory_comparison_3d(
    original: Dict,
    perturbed: Dict,
    title: str = "3D Trajectory Comparison"
) -> None:
    """Plot original vs perturbed trajectory in 3D.
    
    Args:
        original: original trajectory dict
        perturbed: perturbed trajectory dict
        title: plot title
    """
    analyzer_orig = TrajectoryAnalyzer(original)
    analyzer_pert = TrajectoryAnalyzer(perturbed)
    
    ee_orig = analyzer_orig.get_ee_position()
    ee_pert = analyzer_pert.get_ee_position()
    transitions_orig = analyzer_orig.find_gripper_transitions()
    transitions_pert = analyzer_pert.find_gripper_transitions()
    
    if len(ee_orig) == 0:
        print("No EE position found in trajectories")
        return
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(title, fontsize=16)
    
    # Time arrays for color mapping
    orig_time = np.linspace(0, 1, len(ee_orig))
    pert_time = np.linspace(0, 1, len(ee_pert))
    
    # 3D path visualization with color gradient representing time
    # Plot original as a faint grey line, then viridis scatter on top
    ax.plot(ee_orig[:, 0], ee_orig[:, 1], ee_orig[:, 2], color='grey', alpha=0.3, label='Original Path')
    sc_orig = ax.scatter(ee_orig[:, 0], ee_orig[:, 1], ee_orig[:, 2], c=orig_time, cmap='viridis', s=10, alpha=0.8)
    ax.set_zlim(bottom=0)

    # Plot perturbed as a dashed faint line, then plasma scatter with different marker
    ax.plot(ee_pert[:, 0], ee_pert[:, 1], ee_pert[:, 2], color='black', linestyle='--', alpha=0.3, label='Perturbed Path')
    ax.scatter(ee_pert[:, 0], ee_pert[:, 1], ee_pert[:, 2], c=pert_time, cmap='plasma', s=15, marker='x', alpha=0.8)
    
    # Add a colorbar to indicate time passage
    cbar = fig.colorbar(sc_orig, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Passage of Time (Start -> End)')
    
    ax.scatter(ee_orig[0, 0], ee_orig[0, 1], ee_orig[0, 2], c='b', marker='o', s=100, label='Start')
    ax.scatter(ee_orig[-1, 0], ee_orig[-1, 1], ee_orig[-1, 2], c='g', marker='s', s=100, label='End')

    close_orig = [t for t, direction in transitions_orig if direction == 'close' and 0 <= t < len(ee_orig)]
    open_orig = [t for t, direction in transitions_orig if direction == 'open' and 0 <= t < len(ee_orig)]
    close_pert = [t for t, direction in transitions_pert if direction == 'close' and 0 <= t < len(ee_pert)]
    open_pert = [t for t, direction in transitions_pert if direction == 'open' and 0 <= t < len(ee_pert)]

    if close_orig:
        ax.scatter(ee_orig[close_orig, 0], ee_orig[close_orig, 1], ee_orig[close_orig, 2], c='navy', marker='v', s=250, edgecolors='black', linewidths=1.5, depthshade=False, label='Orig Close')
    if open_orig:
        ax.scatter(ee_orig[open_orig, 0], ee_orig[open_orig, 1], ee_orig[open_orig, 2], c='cyan', marker='^', s=250, edgecolors='black', linewidths=1.5, depthshade=False, label='Orig Open')
    if close_pert:
        ax.scatter(ee_pert[close_pert, 0], ee_pert[close_pert, 1], ee_pert[close_pert, 2], c='darkred', marker='v', s=250, edgecolors='black', linewidths=1.5, depthshade=False, label='Pert Close')
    if open_pert:
        ax.scatter(ee_pert[open_pert, 0], ee_pert[open_pert, 1], ee_pert[open_pert, 2], c='orange', marker='^', s=250, edgecolors='black', linewidths=1.5, depthshade=False, label='Pert Open')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

