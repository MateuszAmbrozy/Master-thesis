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
MAX_STEPS = 256
TIMEOUT_CPP = 20.0

# ==========================================

# ==========================================
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


def generate_plots(df, test_name, param_name):
    os.makedirs("plots", exist_ok=True)
    
    try:
        df[param_name] = pd.to_numeric(df[param_name])
    except ValueError:
        pass
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
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
    plt.savefig(f"plots/{test_name}_success_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    time_data = df.groupby([param_name, 'Algorithm'])['Time'].mean()
    time_pivot = time_data.unstack()
    
    ax = time_pivot.plot(kind='bar', figsize=(10, 6), width=0.75, color=colors[:len(time_pivot.columns)])
    plt.title(f'Average Computation Time vs {param_name.capitalize()}', fontsize=16, pad=15)
    plt.ylabel('Time [s] (Log scale)', fontsize=14)
    plt.xlabel(param_name.capitalize(), fontsize=14)
    plt.yscale('log')
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.legend(title='Algorithm', title_fontsize=12, fontsize=11, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"plots/{test_name}_time.png", dpi=300, bbox_inches='tight')
    plt.close()

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
        plt.savefig(f"plots/{test_name}_makespan.png", dpi=300, bbox_inches='tight')
        plt.close()


def run_evaluation(test_name, param_name, param_values, custom_map_path, num_episodes=100):
    for checkpoint_path in [CHECKPOINT_CURRENT, CHECKPOINT_CLASSIC]:
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(checkpoint_path)
    model_current = ACNet_Current(fov_size=FOV_SIZE).to(DEVICE)
    if os.path.exists(CHECKPOINT_CURRENT):
        model_current.load_state_dict(torch.load(CHECKPOINT_CURRENT, map_location=DEVICE)["model"])
        model_current.eval()
    else:
        print(f"Warning: Current checkpoint not found at {CHECKPOINT_CURRENT}")

    model_classic = None
    if ACNet_Classic and os.path.exists(CHECKPOINT_CLASSIC):
        model_classic = ACNet_Classic(fov_size=FOV_SIZE).to(DEVICE)
        model_classic.load_state_dict(torch.load(CHECKPOINT_CLASSIC, map_location=DEVICE)["model"])
        model_classic.eval()
    else:
        print(f"Warning: Classic checkpoint not found at {CHECKPOINT_CLASSIC}")

    
    print(f"Loading map: {custom_map_path}")
    custom_obs_map, map_W, map_H = load_movingai_map(custom_map_path)
    active_grid_size = (map_H, map_W)
    
    raw_results = []
    
    for p_val in param_values:
        save_val = p_val if isinstance(p_val, (int, float)) else str(p_val)
        curr_agents = p_val 

        for ep in range(1, num_episodes + 1):
            shared_seed = np.random.randint(0, 1000000)
            np.random.seed(shared_seed)
            random.seed(shared_seed) 
            
            
            env_curr = MAPFEnv_Current(num_agents=curr_agents, fov_size=FOV_SIZE, SIZE=active_grid_size, max_steps=MAX_STEPS, obstacle_density=(0.0, 0.0))
            starts, goals = generate_starts_and_goals(custom_obs_map, curr_agents)
            env_curr.set_custom_map(custom_obs_map, starts, goals)
            
            
            obs_map = custom_obs_map
            obstacles = [(i, j) for i in range(map_H) for j in range(map_W) if obs_map[i, j] == 1]

            t0 = time.perf_counter()
            cbs_path = cbs_adapter.find_path(obstacles, starts, goals, timeout=TIMEOUT_CPP)
            raw_results.append({param_name: save_val, 'Algorithm': 'CBS', 'Success': cbs_path is not None, 'Makespan': len(cbs_path) if cbs_path else -1, 'Time': time.perf_counter() - t0})

            t0 = time.perf_counter()
            ca_path = ca_adapter.find_path(obstacles, starts, goals, timeout=TIMEOUT_CPP)
            raw_results.append({param_name: save_val, 'Algorithm': 'CA*', 'Success': ca_path is not None, 'Makespan': len(ca_path) if ca_path else -1, 'Time': time.perf_counter() - t0})

            hx, cx = [], []
            for _ in range(curr_agents):
                h, c = model_current.init_state(1, device=DEVICE)
                hx.append(h), cx.append(c)
            steps, done, t0 = 0, False, time.perf_counter()
            prev_acts = [0] * curr_agents
            while not done and steps < MAX_STEPS:
                steps += 1
                positions = env_curr.getAgents()
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
                    obs_maps, goal_vec = env_curr.get_obs(aid)
                    obs_t = torch.from_numpy(np.asarray(obs_maps)).float().to(DEVICE).unsqueeze(0)
                    goal_t = torch.from_numpy(np.asarray(goal_vec)).float().to(DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        policy, _, h_new, c_new, _, _, _ = model_current(obs_t, goal_t, hx[i], cx[i], h_neighbors=h_neighbors, comm_mask=comm_mask, agent_idx=i)
                    hx[i], cx[i] = h_new, c_new
                    valid_acts = env_curr.listNextValidActions(aid, prev_action=prev_acts[i], episode=ep)
                    probs = policy.squeeze(0).detach().cpu().numpy()
                    mask = np.zeros_like(probs, dtype=np.float32)
                    mask[valid_acts] = 1.0
                    masked_probs = probs * mask
                    s = masked_probs.sum()
                    action = int(np.random.choice(valid_acts)) if s <= 1e-12 else int(np.random.choice(np.arange(len(probs)), p=masked_probs/s))
                    prev_acts[i] = action
                    _, _, agent_done, _, _, _, _ = env_curr.step((aid, action))
                    if agent_done: step_done = True
                if step_done: done = True
            raw_results.append({param_name: save_val, 'Algorithm': 'PRIMAL_Improved', 'Success': done, 'Makespan': steps if done else -1, 'Time': time.perf_counter() - t0})

            
            if model_classic:
                np.random.seed(shared_seed)
                env_clas = MAPFEnv_Classic(num_agents=curr_agents, fov_size=FOV_SIZE, SIZE=active_grid_size, max_steps=MAX_STEPS, obstacle_density=(0.0, 0.0))
                env_clas.set_custom_map(custom_obs_map, starts, goals) 
                
                hx_c, cx_c = [], []
                for _ in range(curr_agents):
                    h, c = model_classic.init_state(1, device=DEVICE)
                    hx_c.append(h), cx_c.append(c)
                steps_c, done_c, t0_c = 0, False, time.perf_counter()
                prev_acts_c = [0] * curr_agents
                while not done_c and steps_c < MAX_STEPS:
                    steps_c += 1
                    agent_order = list(range(1, curr_agents + 1))
                    np.random.shuffle(agent_order)
                    step_done_c = False
                    for aid in agent_order:
                        i = aid - 1
                        obs_maps, goal_vec = env_clas.get_obs(aid)
                        obs_t = torch.from_numpy(np.asarray(obs_maps)).float().to(DEVICE).unsqueeze(0)
                        goal_t = torch.from_numpy(np.asarray(goal_vec)).float().to(DEVICE).unsqueeze(0)
                        with torch.no_grad():
                            policy, _, h_new, c_new, _, _, _ = model_classic(obs_t, goal_t, hx_c[i], cx_c[i])
                        hx_c[i], cx_c[i] = h_new, c_new
                        valid_acts = env_clas.listNextValidActions(aid, prev_action=prev_acts_c[i], episode=ep)
                        probs = policy.squeeze(0).detach().cpu().numpy()
                        mask = np.zeros_like(probs, dtype=np.float32)
                        mask[valid_acts] = 1.0
                        masked_probs = probs * mask
                        s = masked_probs.sum()
                        action = 0 if s <= 1e-12 else int(np.random.choice(np.arange(len(masked_probs)), p=masked_probs/s))
                        prev_acts_c[i] = action
                        _, _, agent_done_c, _, _, _, _ = env_clas.step((aid, action))
                        if agent_done_c: step_done_c = True
                    if step_done_c: done_c = True
                raw_results.append({param_name: save_val, 'Algorithm': 'PRIMAL_Classic', 'Success': done_c, 'Makespan': steps_c if done_c else -1, 'Time': time.perf_counter() - t0_c})

            sys.stdout.write(f"\rParam {p_val} | Ep {ep}/{num_episodes}")
            sys.stdout.flush()

            
            temp_df = pd.DataFrame(raw_results)
            temp_df.to_csv(f"partial_{test_name}.csv", index=False)
            
            del env_curr
            if model_classic: del env_clas
            gc.collect()

    df = pd.DataFrame(raw_results)
    df.to_csv(f"raw_{test_name}.csv", index=False)
    
    if os.path.exists(f"partial_{test_name}.csv"):
        os.remove(f"partial_{test_name}.csv")
        
    generate_plots(df, test_name, param_name)
    df_success = df[df['Success'] == True]
    summary_sr = df.groupby([param_name, 'Algorithm'])['Success'].mean() * 100
    summary_time = df.groupby([param_name, 'Algorithm'])['Time'].mean()
    summary_mk = df_success.groupby([param_name, 'Algorithm'])['Makespan'].mean()
    summary = pd.DataFrame({'SR[%]': summary_sr, 'Avg_Mksp': summary_mk, 'Avg_Time[s]': summary_time}).reset_index()
    print(f"\n\nResults: {test_name}\n", summary.to_string(index=False))