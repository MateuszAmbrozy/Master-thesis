import torch
import torch.nn as nn
from ACNet import ACNet
from mapf_env import MAPFEnv
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import os
import sys
import scipy.signal as signal
import matplotlib.pyplot as plt
import time
import random
import pibt_adapter as cpp_mstar

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

torch.set_num_threads(8)
torch.set_flush_denormal(True) 

FOV_SIZE = 10
EPISODE_SAMPLES = 128
NUM_THREADS = 6
PRIMING_LENGTH = 0
DEMONSTRATION_PROB = 0.5
IL_MAX_TRIES = 1000
GRAD_CLIP = 10.0  
MAX_EP = 10000

MIN_AGENTS = 20
MAX_AGENTS = 30

def load_movingai_map(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Map not found: {filepath}")
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
        raise ValueError("Map has insufficient free space for agents!")
    chosen = random.sample(free_cells, 2 * num_agents)
    return chosen[:num_agents], chosen[num_agents:]

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()

class Worker(mp.Process):
    def __init__(
        self, wid, global_net, optimizer, global_ep, res_queue, env_fn,
        min_agents, max_agents, custom_obs_map, fov_size, a_size, gamma=0.95, update_steps=128,      
        entropy_beta=0.01, value_beta=0.5, max_steps_per_ep=600, device="cpu",
    ):
        super().__init__()
        self.wid = wid
        self.name = f"w{wid:02d}"

        self.global_net = global_net
        self.opt = optimizer
        self.global_ep = global_ep
        self.res_queue = res_queue

        self.env = env_fn()
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.custom_obs_map = custom_obs_map
        self.num_agents = max_agents
        self.a_size = a_size

        self.local_net = ACNet(fov_size=fov_size).to(device)
        self.local_net.load_state_dict(self.global_net.state_dict())

        self.gamma = gamma
        self.update_steps = update_steps
        self.entropy_beta = entropy_beta
        self.value_beta = value_beta
        self.max_steps_per_ep = max_steps_per_ep
        self.device = device

    def _prepare_episode(self):
        self.num_agents = np.random.randint(self.min_agents, self.max_agents + 1)
        self.env.num_agents = self.num_agents 

        if self.custom_obs_map is not None:
            starts, goals = generate_starts_and_goals(self.custom_obs_map, self.num_agents)
            self.env.set_custom_map(self.custom_obs_map, starts, goals)
        else:
            self.env.reset()

    def parse_path(self, path):
        result = [[] for _ in range(self.num_agents)]
        total_reward = 0.0

        for t in range(len(path) - 1):
            observations = []
            move_queue = list(range(self.num_agents))
            for agent in range(1, self.num_agents + 1):
                observations.append(self.env.get_obs(agent))

            steps = 0
            while len(move_queue) > 0:
                steps += 1
                i = move_queue.pop(0)

                obs_maps, goal_vec = observations[i]
                pos = path[t][i] 
                newPos = path[t + 1][i]

                direction = (newPos[0] - pos[0], newPos[1] - pos[1])
                a = self.env.world.getAction(direction)

                state, reward, done, nextActions, on_goal, blocking, valid_action = self.env.step((i + 1, a))
                
                if valid_action:
                    total_reward += reward

                if steps > self.num_agents ** 2:
                    return None, 0.0 

                if not valid_action:
                    move_queue.append(i)
                    continue

                result[i].append((obs_maps, goal_vec, a, pos))

        return result, total_reward  

    def _expert_rollouts(self):
        obstacles = self.env.getObstacleMap()
        try:
            starts = [(int(x), int(y)) for x,y in self.env.getAgents()]
            goals  = [(int(x), int(y)) for x,y in self.env.getGoals()]
            mstar_path = cpp_mstar.find_path(obstacles, starts, goals, 30)
        except:
            return None, 0.0, 0

        if mstar_path is None:
            return None, 0.0, 0
        if len(mstar_path) > (self.max_steps_per_ep * 0.6): 
            return None, 0.0, 0

        rollouts, reward = self.parse_path(mstar_path)
        return rollouts, reward, len(mstar_path)
    
    def _update_global_bc(self, rollouts):
        valid_rollouts = [r for r in rollouts if len(r) > 0]
        num_valid = len(valid_rollouts)
        if num_valid == 0: return None

        self.local_net.zero_grad()
        self.opt.zero_grad()

        steps_count = len(valid_rollouts[0])
        total_loss = 0.0

        hx, cx = [], []
        for _ in range(self.num_agents):
            h, c = self.local_net.init_state(1, device=self.device)
            hx.append(h)
            cx.append(c)

        obs_all = torch.tensor(np.array([[step[0] for step in r] for r in valid_rollouts]), dtype=torch.float32, device=self.device)
        goal_all = torch.tensor(np.array([[step[1] for step in r] for r in valid_rollouts]), dtype=torch.float32, device=self.device)
        acts_all = torch.tensor(np.array([[step[2] for step in r] for r in valid_rollouts]), dtype=torch.long, device=self.device)

        for t in range(steps_count):
            h_neighbors = torch.stack([h.detach() for h in hx], dim=1)

            positions = []
            for i in range(self.num_agents):
                if i < num_valid and t < len(valid_rollouts[i]):
                    positions.append(valid_rollouts[i][t][3]) 
                else:
                    positions.append((0, 0))
            
            pos_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)
            pos_mat = (pos_tensor.unsqueeze(1) - pos_tensor.unsqueeze(0)).abs()
            in_obs_mask = (pos_mat <= (self.local_net.fov_size // 2)).all(2)
            comm_mask = (~in_obs_mask).unsqueeze(0)

            agent_order = list(range(num_valid))
            np.random.shuffle(agent_order)

            for i in agent_order:
                obs_t = obs_all[i, t].unsqueeze(0)
                goal_t = goal_all[i, t].unsqueeze(0)
                act_t = acts_all[i, t].unsqueeze(0)

                policy, _, hx_new, cx_new, _, _, _ = self.local_net(
                    obs_t, goal_t, hx[i], cx[i],
                    h_neighbors=h_neighbors,
                    comm_mask=comm_mask,
                    agent_idx=i
                )
                hx[i], cx[i] = hx_new, cx_new

                logp = torch.log(torch.clamp(policy, 1e-10, 1.0))
                loss = F.nll_loss(logp, act_t)
                total_loss = total_loss + loss

        if type(total_loss) is not float:
            total_loss = total_loss / (num_valid * steps_count)
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), GRAD_CLIP)
            
            for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
                if lp.grad is None: continue
                if gp.grad is None:
                    gp.grad = lp.grad.detach().cpu().clone()
                else:
                    gp.grad.copy_(lp.grad.detach().cpu())
            
            self.opt.step()
            return float(total_loss.detach().cpu().item())
            
        return None
    
    def run_il_episode(self, cur_ep):
        self.local_net.load_state_dict(self.global_net.state_dict())
        self._prepare_episode()

        expert_result = self._expert_rollouts()
        if expert_result is None or expert_result[0] is None:
            return 0.0, None, 0

        rollouts, expert_reward, steps = expert_result
        loss = self._update_global_bc(rollouts)

        if loss is None:
            return expert_reward, None, steps

        return expert_reward, loss, steps 

    def _select_action(self, policy_probs, valid_actions):
        probs = policy_probs.detach().cpu().numpy()
        mask = np.zeros_like(probs, dtype=np.float32)
        mask[valid_actions] = 1.0
        masked = probs * mask
        s = masked.sum()
        if s <= 1e-12: return int(np.random.choice(valid_actions))
        masked /= s
        return int(np.random.choice(np.arange(len(probs)), p=masked))
    
    def run_rl_episode(self, cur_ep):
        self.local_net.load_state_dict(self.global_net.state_dict())
        self._prepare_episode()

        hx, cx = [], []
        for _ in range(self.num_agents):
            h, c = self.local_net.init_state(1, device=self.device)
            hx.append(h)
            cx.append(c)

        agent_buffers = [[] for _ in range(self.num_agents)]

        ep_reward = 0.0
        done = False
        steps = 0
        prev_actions = [0] * self.num_agents

        while (not done) and (steps < self.max_steps_per_ep):
            steps += 1

            positions = self.env.getAgents()
            pos_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)
            pos_mat = (pos_tensor.unsqueeze(1) - pos_tensor.unsqueeze(0)).abs()
            in_obs_mask = (pos_mat <= (self.local_net.fov_size // 2)).all(2)
            comm_mask = (~in_obs_mask).unsqueeze(0) 
            
            h_neighbors = torch.stack([h.detach() for h in hx], dim=1)

            agent_order = list(range(1, self.num_agents + 1))
            np.random.shuffle(agent_order)

            for aid in agent_order:
                i = aid - 1 

                obs_maps, goal_vec = self.env.get_obs(aid)
                valid = self.env.listNextValidActions(aid, prev_action=prev_actions[i], episode=cur_ep)
                
                valid_onehot = np.zeros(self.a_size, dtype=np.float32)
                valid_onehot[valid] = 1.0

                obs_t = torch.from_numpy(np.asarray(obs_maps)).float().to(self.device).unsqueeze(0)
                goal_t = torch.from_numpy(np.asarray(goal_vec)).float().to(self.device).unsqueeze(0)

                policy, value, hx_new, cx_new, blocking_pred, on_goal_pred, policy_sig = self.local_net(
                    obs_t, goal_t, hx[i], cx[i],
                    h_neighbors=h_neighbors,
                    comm_mask=comm_mask,
                    agent_idx=i
                )
                hx[i], cx[i] = hx_new, cx_new

                p = policy.squeeze(0)

                a = self._select_action(p, valid)
                prev_actions[i] = a

                logp = torch.log(torch.clamp(p[a], 1e-10, 1.0))
                ent = -(p * torch.log(torch.clamp(p, 1e-10, 1.0))).sum()

                state, reward, step_done, _, _, blocking_env, _ = self.env.step((aid, a))

                ep_reward += float(reward)

                agent_buffers[i].append({
                    'value': value.squeeze(0).squeeze(-1),
                    'logp': logp,
                    'ent': ent,
                    'reward': float(reward),
                    'policy_sig': policy_sig.squeeze(0),
                    'train_valid': torch.tensor(valid_onehot, dtype=torch.float32, device=self.device),
                    'blocking_pred': blocking_pred.squeeze(0).squeeze(-1),
                    'target_blocking': torch.tensor(1.0 if blocking_env else 0.0, dtype=torch.float32, device=self.device)
                })

                if step_done:
                    done = True

            if done:
                for b in agent_buffers:
                    if len(b) > 0:
                        b[-1]['reward'] += 20.0
                ep_reward += (20.0 * self.num_agents) 

            if (steps % self.update_steps == 0) or done:
                self._update_global_rl(agent_buffers, done, hx, cx)
                hx = [h.detach() for h in hx]
                cx = [c.detach() for c in cx]
                agent_buffers = [[] for _ in range(self.num_agents)]
                self.local_net.load_state_dict(self.global_net.state_dict())

        return ep_reward, steps

    def _update_global_rl(self, agent_buffers, done, hx, cx):
            valid_agents = sum(1 for b in agent_buffers if len(b) > 0)
            if valid_agents == 0: return

            self.local_net.zero_grad()
            self.opt.zero_grad()

            R_boots = [0.0] * self.num_agents
            if not done:
                with torch.no_grad():
                    for i, aid in enumerate(range(1, self.num_agents + 1)):
                        if len(agent_buffers[i]) == 0: continue
                        obs_maps, goal_vec = self.env.get_obs(aid)
                        obs_t  = torch.from_numpy(np.asarray(obs_maps)).float().to(self.device).unsqueeze(0)
                        goal_t = torch.from_numpy(np.asarray(goal_vec)).float().to(self.device).unsqueeze(0)
                        
                        _, value, _, _, _, _, _ = self.local_net(obs_t, goal_t, hx[i], cx[i])
                        R_boots[i] = float(value.squeeze().item())

            total_loss = 0.0

            for i in range(self.num_agents):
                trajectory = agent_buffers[i]
                if len(trajectory) == 0: continue
                    
                R = R_boots[i]
                agent_returns = []
                
                for step_data in reversed(trajectory):
                    R = step_data['reward'] + self.gamma * R
                    agent_returns.append(R)
                
                agent_returns.reverse()
                
                returns_t = torch.tensor(agent_returns, dtype=torch.float32, device=self.device)
                values_t = torch.stack([step['value'] for step in trajectory])
                logps_t = torch.stack([step['logp'] for step in trajectory])
                ents_t = torch.stack([step['ent'] for step in trajectory])
                
                policy_sigs_t = torch.stack([step['policy_sig'] for step in trajectory])
                train_valids_t = torch.stack([step['train_valid'] for step in trajectory])
                blocking_preds_t = torch.stack([step['blocking_pred'] for step in trajectory])
                target_blockings_t = torch.stack([step['target_blocking'] for step in trajectory])

                adv = returns_t - values_t.detach()
                if adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                else:
                    adv = adv - adv.mean()

                policy_loss = -(logps_t * adv).mean()
                value_loss = (returns_t - values_t).pow(2).mean()
                entropy_loss = ents_t.mean()
                
                bce_full = F.binary_cross_entropy(policy_sigs_t, train_valids_t, reduction='none')
                invalid_mask = 1.0 - train_valids_t
                valid_loss = (bce_full * invalid_mask).sum() / (invalid_mask.sum() + 1e-8)

                blocking_loss = F.binary_cross_entropy(blocking_preds_t, target_blockings_t.float(), reduction='mean')

                loss = policy_loss + (0.5 * value_loss) + (0.5 * valid_loss) + (0.5 * blocking_loss) - (self.entropy_beta * entropy_loss)
                total_loss = total_loss + loss
            
            if type(total_loss) is not float:
                total_loss = total_loss / valid_agents
                total_loss.backward()
            
                torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), GRAD_CLIP)

                for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
                    if lp.grad is None: continue
                    if gp.grad is None:
                        gp.grad = lp.grad.detach().cpu().clone()
                    else:
                        gp.grad.copy_(lp.grad.detach().cpu())

                self.opt.step()

    def run(self):
            torch.set_num_threads(1)
            while True:
                with self.global_ep.get_lock():
                    if self.global_ep.value >= MAX_EP:
                        break
                    cur_ep = int(self.global_ep.value)

                use_il = (cur_ep < PRIMING_LENGTH) or (np.random.rand() < DEMONSTRATION_PROB)

                if use_il:
                    ep_reward, bc_loss, steps = self.run_il_episode(cur_ep)
                    mode = "IL"
                    success = (steps > 0)
                else:
                    ep_reward, steps = self.run_rl_episode(cur_ep)
                    bc_loss = None
                    mode = "RL"
                    success = (steps < self.max_steps_per_ep)

                with self.global_ep.get_lock():
                    self.global_ep.value += 1
                    ep = int(self.global_ep.value)
                    
                current_density = self.env.really_obstacles_per if hasattr(self.env, "really_obstacles_per") and self.env.really_obstacles_per is not None else 0.0
                map_shape = self.env.world.state.shape
                
                self.res_queue.put((ep, ep_reward, mode, bc_loss, steps, current_density, map_shape, success, self.num_agents))

            self.res_queue.put(None)

def make_env():
    return MAPFEnv(num_agents=MAX_AGENTS, fov_size=10, SIZE=(20,40), max_steps=600, obstacle_density=(0.3, 0.5))

if __name__ == "__main__":
    custom_map_path = None
    custom_obs_map = None
    
    if len(sys.argv) > 1:
        custom_map_path = sys.argv[1]
        try:
            custom_obs_map, map_W, map_H = load_movingai_map(custom_map_path)
            print(f"[{'='*40}]")
            print(f"CUSTOM MODE: Training on loaded map {custom_map_path} ({map_W}x{map_H})")
            print(f"Agents per episode: from {MIN_AGENTS} to {MAX_AGENTS}")
            print(f"[{'='*40}]")
        except Exception as e:
            print(f"Error loading map: {e}")
            sys.exit(1)
    else:
        print(f"[{'='*40}]")
        print(f"DEFAULT MODE: Training on randomly generated maps.")
        print(f"Agents per episode: from {MIN_AGENTS} to {MAX_AGENTS}")
        print(f"[{'='*40}]")

    seed = int(time.time())
    np.random.seed(seed)
    mp.set_start_method("spawn", force=True)
    gnet = ACNet(fov_size=10)
    gnet.share_memory()
    
    opt = SharedAdam(gnet.parameters(), lr=1e-5, betas=(0.92, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    
    CHECKPOINT_PATH = "checkpoint_warehouse.pt"
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Saved model found: {CHECKPOINT_PATH}. Loading...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        gnet.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        # global_ep.value = checkpoint.get("episode", 0)
        global_ep.value = 0
        print(f"Resumed training from episode: {global_ep.value}")

    workers = [Worker( 
        wid=i, 
        global_net=gnet, 
        optimizer=opt, 
        global_ep=global_ep, 
        res_queue=res_queue, 
        env_fn=make_env, 
        min_agents=MIN_AGENTS, 
        max_agents=MAX_AGENTS, 
        custom_obs_map=custom_obs_map, 
        fov_size=10, 
        a_size=5, 
        device="cpu" 
    ) for i in range(NUM_THREADS)]
    
    [w.start() for w in workers]
    episodes = []
    rewards = []
    successes_100 = []

    try:
        while True:
            r = res_queue.get()
            if r is None:
                break
                
            ep, ep_reward, mode, bc_loss, steps, obs_dens, map_shape, success, num_active_agents = r
            
            mean_reward = np.mean(rewards[-100:]) if len(rewards) >= 1 else ep_reward
            
            successes_100.append(1.0 if success else 0.0)
            if len(successes_100) > 100:
                successes_100.pop(0)
            sr_100 = np.mean(successes_100) * 100 if len(successes_100) > 0 else 0.0

            loss_str = f"{bc_loss:.4f}" if bc_loss is not None else " N/A  "
            
            if custom_obs_map is not None:
                dens_str = "CUST" 
            else:
                dens_str = f"{obs_dens*100:2.0f}%"

            print(
                f"[EP {ep:7d}] "
                f"Mode: {mode} | "
                f"SR(100): {sr_100:5.1f}% | "
                f"Ag: {num_active_agents:2d} | "
                f"Map: {map_shape[0]}x{map_shape[1]} | "
                f"Dens: {dens_str:>4} | "
                f"Steps: {steps:4d} | "
                f"Rwd: {ep_reward:7.2f} | "
                f"Mean(100): {mean_reward:7.2f} | "
                f"IL_Loss: {loss_str}"
            )
            
            episodes.append(ep)
            rewards.append(ep_reward)

            if ep > 0 and ep % 1000 == 0:
                print(f">>> Automatic save after {ep} episodes...")
                torch.save(
                    {
                        "model": gnet.state_dict(),
                        "optimizer": opt.state_dict(),
                        "episode": ep
                    },
                    "checkpoint_warehouse.pt"
                )

    except KeyboardInterrupt:
            current_ep = global_ep.value
            print(f"\n[Ctrl+C] Training interrupted at episode {current_ep}! Saving emergency model checkpoint...")
            
            torch.save(
                {
                    "model": gnet.state_dict(),
                    "optimizer": opt.state_dict(),
                    "episode": current_ep
                },
                "checkpoint_warehouse.pt"
            )
            print(">>> Emergency checkpoint saved successfully.")

            for w in workers:
                w.terminate()

    finally:
            for w in workers:
                w.join()
            
            if len(episodes) > 0:
                plt.plot(episodes, rewards)
                plt.show()