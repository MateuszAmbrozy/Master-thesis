import sys
import os
import time
import random
import gc
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "pyproject.toml").exists())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from primal_mapf.classic.network import ACNet as ACNet_Classic
from primal_mapf.classic.environment import MAPFEnv as MAPFEnv_Classic
from primal_mapf.improved.network import ACNet as ACNet_Current
from primal_mapf.improved.environment import MAPFEnv as MAPFEnv_Current
from primal_mapf.solvers.api import find_path as solve_path


def _historical_solver(name, obstacles, starts, goals, timeout=20.0):
    """Bridge the archived coordinate-list interface to the shared solvers."""
    coordinates = list(obstacles) + list(starts) + list(goals)
    shape = tuple(max(int(point[axis]) for point in coordinates) + 1 for axis in [0, 1])
    grid = np.zeros(shape, dtype=int)
    for row, column in obstacles:
        grid[row, column] = 1
    return solve_path(name, grid, starts, goals, timeout=timeout)


ca_adapter = SimpleNamespace(find_path=lambda *args, **kwargs: _historical_solver("ca", *args, **kwargs))
cbs_adapter = SimpleNamespace(find_path=lambda *args, **kwargs: _historical_solver("cbs", *args, **kwargs))
CHECKPOINT_CURRENT = str(PROJECT_ROOT / "checkpoints/base/improved.pt")
CHECKPOINT_CLASSIC = str(PROJECT_ROOT / "checkpoints/base/classic.pt")

DEVICE = "cpu"
FOV_SIZE = 10
MAX_STEPS = 500

def load_movingai_map(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    height = 0; width = 0; map_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("height"): height = int(line.split()[1])
        elif line.startswith("width"): width = int(line.split()[1])
        elif line.startswith("map"):
            map_start = i + 1
            break
            
    obs_map = np.zeros((height, width), dtype=int)
    passable = {'.', 'G'}
    
    for r in range(height):
        row = lines[map_start + r].strip()
        for c in range(width):
            if row[c] not in passable:
                obs_map[r, c] = 1
                
    return obs_map, width, height

def generate_starts_and_goals(obs_map, num_agents):
    H, W = obs_map.shape
    free_cells = [(r, c) for r in range(H) for c in range(W) if obs_map[r, c] == 0]
    
    if len(free_cells) < 2 * num_agents:
        raise ValueError("Map has insufficient free space for this many agents!")
        
    chosen = random.sample(free_cells, 2 * num_agents)
    return chosen[:num_agents], chosen[num_agents:]
def generate_plots(df, test_name, param_name, output_dir="."):
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        df[param_name] = pd.to_numeric(df[param_name])
    except ValueError:
        pass
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    # --- Success rate plot ---
    plt.figure(figsize=(10, 6))
    sr_data = df.groupby([param_name, 'Algorithm'])['Success'].mean() * 100
    sr_pivot = sr_data.unstack()
    
    for i, col in enumerate(sr_pivot.columns):
        plt.plot(sr_pivot.index, sr_pivot[col], marker=markers[i%len(markers)], color=colors[i%len(colors)], 
                 label=col, linewidth=2.5, markersize=8)
        
    plt.title(f'Success Rate vs {param_name.capitalize()}', fontsize=16, pad=15)
    plt.ylabel('Success Rate [%]', fontsize=14)
    plt.xlabel(param_name.capitalize(), fontsize=14)
    plt.ylim(-5, 105)
    plt.xticks(sr_pivot.index, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Algorithm', title_fontsize=12, fontsize=11, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{test_name}_success_rate.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Runtime plot ---
    plt.figure(figsize=(10, 6))
    time_data = df.groupby([param_name, 'Algorithm'])['Time'].mean()
    time_pivot = time_data.unstack()
    ax = time_pivot.plot(kind='bar', figsize=(10, 6), width=0.5, color=colors[:len(time_pivot.columns)])
    plt.title(f'Average Computation Time vs {param_name.capitalize()}', fontsize=16, pad=15)
    plt.ylabel('Time [s]', fontsize=14)
    plt.xlabel(param_name.capitalize(), fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Algorithm', title_fontsize=12, fontsize=11, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{test_name}_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Makespan plot ---
    df_succ = df[df['Success'] == True]
    if not df_succ.empty:
        plt.figure(figsize=(10, 6))
        mk_data = df_succ.groupby([param_name, 'Algorithm'])['Makespan'].mean()
        mk_pivot = mk_data.unstack()
        
        for i, col in enumerate(mk_pivot.columns):
            plt.plot(mk_pivot.index, mk_pivot[col], marker=markers[i%len(markers)], color=colors[i%len(colors)], 
                     label=col, linewidth=2.5, markersize=8)
            
        plt.title(f'Mean Makespan vs {param_name.capitalize()} (Successful Episodes)', fontsize=16, pad=15)
        plt.ylabel('Steps', fontsize=14)
        plt.xlabel(param_name.capitalize(), fontsize=14)
        plt.xticks(mk_pivot.index, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Algorithm', title_fontsize=12, fontsize=11, bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{test_name}_makespan.png"), dpi=300, bbox_inches='tight')
        plt.close()


def run_evaluation(test_name, param_name, param_values, custom_map_path, model_ft_path, model_base_path, use_current_architecture, num_episodes=20):
    if use_current_architecture:
        ActiveACNet = ACNet_Current
        ActiveMAPFEnv = MAPFEnv_Current
    else:
        if ACNet_Classic is None:
            print("ERROR: Classic architecture (classic_primal) not found!")
            return
        ActiveACNet = ACNet_Classic
        ActiveMAPFEnv = MAPFEnv_Classic

    model_ft = ActiveACNet(fov_size=FOV_SIZE).to(DEVICE)
    if os.path.exists(model_ft_path):
        model_ft.load_state_dict(torch.load(model_ft_path, map_location=DEVICE)["model"])
        model_ft.eval()
    else:
        print(f"ERROR: Missing fine-tuned checkpoint: {model_ft_path}")
        return

    model_base = ActiveACNet(fov_size=FOV_SIZE).to(DEVICE)
    if os.path.exists(model_base_path):
        model_base.load_state_dict(torch.load(model_base_path, map_location=DEVICE)["model"])
        model_base.eval()
    else:
        print(f"ERROR: Missing baseline checkpoint: {model_base_path}")
        return

    custom_obs_map, map_W, map_H = load_movingai_map(custom_map_path)
    active_grid_size = (map_H, map_W)
    
    raw_results = []
    
    for p_val in param_values:
        save_val = p_val if isinstance(p_val, (int, float)) else str(p_val)
        curr_agents = p_val 

        for ep in range(1, num_episodes + 1):
            shared_seed = np.random.randint(0, 1000000)
            
            random.seed(shared_seed)
            starts, goals = generate_starts_and_goals(custom_obs_map, curr_agents)
            
            # --- WARIANT 1: PRIMAL_FineTuned ---
            np.random.seed(shared_seed)
            env_ft = ActiveMAPFEnv(num_agents=curr_agents, fov_size=FOV_SIZE, SIZE=active_grid_size, max_steps=MAX_STEPS, obstacle_density=(0.0, 0.0))
            env_ft.set_custom_map(custom_obs_map, starts, goals)
            
            hx, cx = [], []
            for _ in range(curr_agents):
                h, c = model_ft.init_state(1, device=DEVICE)
                hx.append(h), cx.append(c)
                
            steps, done, t0 = 0, False, time.perf_counter()
            prev_acts = [0] * curr_agents
            
            while not done and steps < MAX_STEPS:
                steps += 1
                positions = env_ft.getAgents()
                
                if use_current_architecture:
                    pos_tensor = torch.tensor(positions, dtype=torch.float32, device=DEVICE)
                    pos_mat = (pos_tensor.unsqueeze(1) - pos_tensor.unsqueeze(0)).abs()
                    in_obs_mask = (pos_mat <= (FOV_SIZE // 2)).all(2)
                    comm_mask = (~in_obs_mask).unsqueeze(0)
                    h_neighbors = torch.stack([h.detach() for h in hx], dim=1)
                
                agent_order = list(range(1, curr_agents + 1))
                np.random.shuffle(agent_order)
                step_done = False
                
                for aid in agent_order:
                    i = aid - 1
                    obs_maps, goal_vec = env_ft.get_obs(aid)
                    obs_t = torch.from_numpy(np.asarray(obs_maps)).float().to(DEVICE).unsqueeze(0)
                    goal_t = torch.from_numpy(np.asarray(goal_vec)).float().to(DEVICE).unsqueeze(0)
                    
                    with torch.no_grad():
                        if use_current_architecture:
                            policy, _, h_new, c_new, _, _, _ = model_ft(obs_t, goal_t, hx[i], cx[i], h_neighbors=h_neighbors, comm_mask=comm_mask, agent_idx=i)
                        else:
                            policy, _, h_new, c_new, _, _, _ = model_ft(obs_t, goal_t, hx[i], cx[i])
                            
                    hx[i], cx[i] = h_new, c_new
                    valid_acts = env_ft.listNextValidActions(aid, prev_action=prev_acts[i], episode=ep)
                    probs = policy.squeeze(0).detach().cpu().numpy()
                    mask = np.zeros_like(probs, dtype=np.float32)
                    mask[valid_acts] = 1.0
                    masked_probs = probs * mask
                    s = masked_probs.sum()
                    
                    action = int(np.random.choice(valid_acts)) if s <= 1e-12 else int(np.random.choice(np.arange(len(probs)), p=masked_probs/s))
                    prev_acts[i] = action
                    _, _, agent_done, _, _, _, _ = env_ft.step((aid, action))
                    if agent_done: step_done = True
                if step_done: done = True
                
            raw_results.append({param_name: save_val, 'Algorithm': 'PRIMAL_FineTuned', 'Success': done, 'Makespan': steps if done else -1, 'Time': time.perf_counter() - t0})

            # --- WARIANT 2: PRIMAL_Baseline ---
            np.random.seed(shared_seed)
            env_base = ActiveMAPFEnv(num_agents=curr_agents, fov_size=FOV_SIZE, SIZE=active_grid_size, max_steps=MAX_STEPS, obstacle_density=(0.0, 0.0))
            env_base.set_custom_map(custom_obs_map, starts, goals)
            
            hx_b, cx_b = [], []
            for _ in range(curr_agents):
                h, c = model_base.init_state(1, device=DEVICE)
                hx_b.append(h), cx_b.append(c)
                
            steps_b, done_b, t0_b = 0, False, time.perf_counter()
            prev_acts_b = [0] * curr_agents
            
            while not done_b and steps_b < MAX_STEPS:
                steps_b += 1
                positions_b = env_base.getAgents()
                
                if use_current_architecture:
                    pos_tensor_b = torch.tensor(positions_b, dtype=torch.float32, device=DEVICE)
                    pos_mat_b = (pos_tensor_b.unsqueeze(1) - pos_tensor_b.unsqueeze(0)).abs()
                    in_obs_mask_b = (pos_mat_b <= (FOV_SIZE // 2)).all(2)
                    comm_mask_b = (~in_obs_mask_b).unsqueeze(0)
                    h_neighbors_b = torch.stack([h.detach() for h in hx_b], dim=1)
                
                agent_order = list(range(1, curr_agents + 1))
                np.random.shuffle(agent_order)
                step_done_b = False
                
                for aid in agent_order:
                    i = aid - 1
                    obs_maps, goal_vec = env_base.get_obs(aid)
                    obs_t = torch.from_numpy(np.asarray(obs_maps)).float().to(DEVICE).unsqueeze(0)
                    goal_t = torch.from_numpy(np.asarray(goal_vec)).float().to(DEVICE).unsqueeze(0)
                    
                    with torch.no_grad():
                        if use_current_architecture:
                            policy, _, h_new, c_new, _, _, _ = model_base(obs_t, goal_t, hx_b[i], cx_b[i], h_neighbors=h_neighbors_b, comm_mask=comm_mask_b, agent_idx=i)
                        else:
                            policy, _, h_new, c_new, _, _, _ = model_base(obs_t, goal_t, hx_b[i], cx_b[i])
                            
                    hx_b[i], cx_b[i] = h_new, c_new
                    valid_acts = env_base.listNextValidActions(aid, prev_action=prev_acts_b[i], episode=ep)
                    probs = policy.squeeze(0).detach().cpu().numpy()
                    mask = np.zeros_like(probs, dtype=np.float32)
                    mask[valid_acts] = 1.0
                    masked_probs = probs * mask
                    s = masked_probs.sum()
                    
                    action = int(np.random.choice(valid_acts)) if s <= 1e-12 else int(np.random.choice(np.arange(len(probs)), p=masked_probs/s))
                    prev_acts_b[i] = action
                    _, _, agent_done_b, _, _, _, _ = env_base.step((aid, action))
                    if agent_done_b: step_done_b = True
                if step_done_b: done_b = True
                
            raw_results.append({param_name: save_val, 'Algorithm': 'PRIMAL_Baseline', 'Success': done_b, 'Makespan': steps_b if done_b else -1, 'Time': time.perf_counter() - t0_b})

            sys.stdout.write(f"\rImproved architecture: {use_current_architecture} | Agents {p_val} | Ep {ep}/{num_episodes}")
            sys.stdout.flush()


            output_folder = f"results_{test_name.split('_')[-1]}" 
            os.makedirs(output_folder, exist_ok=True)
            
            temp_df = pd.DataFrame(raw_results)
            temp_df.to_csv(os.path.join(output_folder, f"partial_{test_name}.csv"), index=False)
            
            del env_ft, env_base
            gc.collect()

    df = pd.DataFrame(raw_results)
    df.to_csv(os.path.join(output_folder, f"raw_{test_name}.csv"), index=False)
    
    if os.path.exists(os.path.join(output_folder, f"partial_{test_name}.csv")):
        os.remove(os.path.join(output_folder, f"partial_{test_name}.csv"))
        
    generate_plots(df, test_name, param_name, output_dir=output_folder)