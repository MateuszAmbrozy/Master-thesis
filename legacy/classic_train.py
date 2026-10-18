import torch
import torch.nn as nn
from ACNet import ACNet
from mapf_env import MAPFEnv
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import gymnasium as gym
import os
import scipy.signal as signal
import matplotlib.pyplot as plt
import cbs_adapter as cpp_mstar

os.environ["OMP_NUM_THREADS"] = "1"


FOV_SIZE = 10
EPISODE_SAMPLES = 128
NUM_THREADS = 8
PRIMING_LENGTH = 0
DEMONSTRATION_PROB = 0.5       
IL_MAX_TRIES = 1               
GRAD_CLIP = 1000.0               
MAX_EP = 3_000_000

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def good_discount(x, gamma):
    return discount(x,gamma)

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
        self,
        wid: int,
        global_net: ACNet,
        optimizer: SharedAdam,
        global_ep: mp.Value,
        res_queue: mp.Queue,
        env_fn,
        num_agents: int,
        fov_size: int,
        a_size: int,
        gamma: float = 0.95,          
        update_steps: int = 128,      
        entropy_beta: float = 0.03,   
        value_beta: float = 0.5,
        max_steps_per_ep: int = 256,  
        device: str = "cpu",
    ):
        super().__init__()
        self.wid = wid
        self.name = f"w{wid:02d}"

        self.global_net = global_net
        self.opt = optimizer
        self.global_ep = global_ep
        self.res_queue = res_queue

        self.env = env_fn()
        self.num_agents = num_agents
        self.a_size = a_size

        self.local_net = ACNet(fov_size=fov_size).to(device)
        self.local_net.load_state_dict(self.global_net.state_dict())

        self.gamma = gamma
        self.update_steps = update_steps
        self.entropy_beta = entropy_beta
        self.value_beta = value_beta
        self.max_steps_per_ep = max_steps_per_ep
        self.device = device

    # ---------------- IL (BC) ----------------
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

                result[i].append((obs_maps, goal_vec, a))

        return result, total_reward  

    def _expert_rollouts(self):
        obstacles = self.env.getObstacleMap()
        try:
            starts = [(int(x), int(y)) for x,y in self.env.getAgents()]
            goals  = [(int(x), int(y)) for x,y in self.env.getGoals()]
            mstar_path = cpp_mstar.find_path(obstacles, starts, goals, 5)
        except:
            return None, 0.0, 0

        if mstar_path is None:
            return None, 0.0, 0

        rollouts, reward = self.parse_path(mstar_path)
        return rollouts, reward, len(mstar_path)
    
    def _update_global_bc(self, rollout):
            if len(rollout) == 0:
                return None

            obs = torch.tensor(np.stack([r[0] for r in rollout]), dtype=torch.float32, device=self.device)
            goal = torch.tensor(np.stack([r[1] for r in rollout]), dtype=torch.float32, device=self.device)
            acts = torch.tensor([r[2] for r in rollout], dtype=torch.long, device=self.device)

            
            hx, cx = self.local_net.init_state(1, device=self.device)
            
            logps = []
            
            for t in range(obs.size(0)):
                
                policy, value, hx_new, cx_new, blocking_pred, on_goal_pred, policy_sig = self.local_net(
                    obs[t].unsqueeze(0), goal[t].unsqueeze(0), hx, cx
                )
                hx, cx = hx_new, cx_new
                logps.append(torch.log(torch.clamp(policy.squeeze(0), 1e-10, 1.0)))

            logps_t = torch.stack(logps)
            
            
            bc_loss = F.nll_loss(logps_t, acts, reduction='mean')

            self.opt.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), GRAD_CLIP)

            for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
                if lp.grad is None: continue
                if gp.grad is None:
                    gp.grad = lp.grad.detach().cpu().clone()
                else:
                    gp.grad.copy_(lp.grad.detach().cpu())

            self.opt.step()
            return float(bc_loss.detach().cpu().item())
    
    def run_il_episode(self, cur_ep):
        self.local_net.load_state_dict(self.global_net.state_dict())
        self.env.reset()

        expert_result = self._expert_rollouts()
        if expert_result is None or expert_result[0] is None:
            return 0.0, None, 0

        rollouts, expert_reward, steps = expert_result
        losses = []
        for agent_rollout in rollouts:
            if len(agent_rollout) == 0: continue
            loss = self._update_global_bc(agent_rollout)
            if loss is not None:
                losses.append(loss)

        if len(losses) == 0:
            return expert_reward, None, steps

        return expert_reward, float(np.mean(losses)), steps 

    # ---------------- RL (A3C) ----------------
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
        self.env.reset()
        self.local_net.load_state_dict(self.global_net.state_dict())

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

            for i, aid in enumerate(range(1, self.num_agents + 1)):

                obs_maps, goal_vec = self.env.get_obs(aid)
                valid = self.env.listNextValidActions(
                    aid,
                    prev_action=prev_actions[i],
                    episode=cur_ep
                )
                
                
                valid_onehot = np.zeros(self.a_size, dtype=np.float32)
                valid_onehot[valid] = 1.0

                obs_t = torch.from_numpy(np.asarray(obs_maps)).float().to(self.device).unsqueeze(0)
                goal_t = torch.from_numpy(np.asarray(goal_vec)).float().to(self.device).unsqueeze(0)

                
                policy, value, hx_new, cx_new, blocking_pred, on_goal_pred, policy_sig = self.local_net(obs_t, goal_t, hx[i], cx[i])
                hx[i], cx[i] = hx_new, cx_new

                p = policy.squeeze(0)

                a = self._select_action(p, valid)
                prev_actions[i] = a

                logp = torch.log(torch.clamp(p[a], 1e-10, 1.0))
                ent = -(p * torch.log(torch.clamp(p, 1e-10, 1.0))).sum()

                
                state, reward, done, _, _, blocking_env, _ = self.env.step((aid, a))

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

                if done:
                    
                    for b in agent_buffers:
                        if len(b) > 0:
                            b[-1]['reward'] += 20.0
                    ep_reward += (20.0 * self.num_agents) 
                    break

            if (steps % self.update_steps == 0) or done:
                self._update_global_rl(agent_buffers, done, hx, cx)
                hx = [h.detach() for h in hx]
                cx = [c.detach() for c in cx]
                agent_buffers = [[] for _ in range(self.num_agents)]
                self.local_net.load_state_dict(self.global_net.state_dict())

        return ep_reward, steps

    def _update_global_rl(self, agent_buffers, done, hx, cx):
            if sum(len(b) for b in agent_buffers) == 0:
                return

            R_boots = [0.0] * self.num_agents
            if not done:
                with torch.no_grad():
                    for i, aid in enumerate(range(1, self.num_agents + 1)):
                        obs_maps, goal_vec = self.env.get_obs(aid)
                        obs_t  = torch.from_numpy(np.asarray(obs_maps)).float().to(self.device).unsqueeze(0)
                        goal_t = torch.from_numpy(np.asarray(goal_vec)).float().to(self.device).unsqueeze(0)
                        
                        
                        _, value, _, _, _, _, _ = self.local_net(obs_t, goal_t, hx[i], cx[i])
                        R_boots[i] = float(value.squeeze().item())

            all_returns, all_values, all_logps, all_ents = [], [], [], []
            
            all_policy_sigs, all_train_valids = [], []
            all_blocking_preds, all_target_blockings = [], []

            for i in range(self.num_agents):
                trajectory = agent_buffers[i]
                if len(trajectory) == 0: continue
                    
                R = R_boots[i]
                agent_returns = []
                
                for step_data in reversed(trajectory):
                    R = step_data['reward'] + self.gamma * R
                    agent_returns.append(R)
                
                agent_returns.reverse()
                
                all_returns.extend(agent_returns)
                all_values.extend([step['value'] for step in trajectory])
                all_logps.extend([step['logp'] for step in trajectory])
                all_ents.extend([step['ent'] for step in trajectory])
                
                
                all_policy_sigs.extend([step['policy_sig'] for step in trajectory])
                all_train_valids.extend([step['train_valid'] for step in trajectory])
                all_blocking_preds.extend([step['blocking_pred'] for step in trajectory])
                all_target_blockings.extend([step['target_blocking'] for step in trajectory])

            returns_t = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
            values_t = torch.stack(all_values)
            logps_t = torch.stack(all_logps)
            ents_t = torch.stack(all_ents)
            
            
            policy_sigs_t = torch.stack(all_policy_sigs)
            train_valids_t = torch.stack(all_train_valids)
            blocking_preds_t = torch.stack(all_blocking_preds)
            target_blockings_t = torch.stack(all_target_blockings)

            adv = returns_t - values_t.detach()

            
            policy_loss = -(logps_t * adv).sum()
            value_loss = (returns_t - values_t).pow(2).sum()
            entropy_loss = ents_t.sum()
            
            
            valid_loss = F.binary_cross_entropy(policy_sigs_t, train_valids_t, reduction='sum')
            
            
            blocking_loss = F.binary_cross_entropy(
                blocking_preds_t, 
                target_blockings_t.float(), 
                reduction='sum'
            )

            
            loss = policy_loss + (0.5 * value_loss) + (0.5 * valid_loss) + (0.5 * blocking_loss) - (self.entropy_beta * entropy_loss)

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), GRAD_CLIP)

            for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
                if lp.grad is None: continue
                if gp.grad is None:
                    gp.grad = lp.grad.detach().cpu().clone()
                else:
                    gp.grad.copy_(lp.grad.detach().cpu())

            self.opt.step()

    # ---------------- main run ----------------
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
            else:
                ep_reward, steps = self.run_rl_episode(cur_ep)
                bc_loss = None
                mode = "RL"

            with self.global_ep.get_lock():
                self.global_ep.value += 1
                ep = int(self.global_ep.value)
            current_density = self.env.really_obstacles_per
            self.res_queue.put((ep, ep_reward, mode, bc_loss, steps, current_density))


        self.res_queue.put(None)

def make_env():
    
    return MAPFEnv(num_agents=4, fov_size=10, SIZE=(15,15), max_steps=256, obstacle_density=(0.1, 0.2))
import time
if __name__ == "__main__":
    seed = int(time.time())
    np.random.seed(seed)
    print("seed =", seed)
    mp.set_start_method("spawn", force=True)
    gnet = ACNet(fov_size=10)
    gnet.share_memory()
    
    
    opt = SharedAdam(gnet.parameters(), lr=2e-5, betas=(0.92, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    
    CHECKPOINT_PATH = "checkpoint.pt"
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Saved model found: {CHECKPOINT_PATH}. Loading...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        gnet.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        global_ep.value = 0
        print(f"Resumed training from episode: {global_ep.value}")
    else:
        print("No checkpoint found. Starting training from scratch (episode 0).")

    workers = [Worker( 
        wid=i, 
        global_net=gnet, 
        optimizer=opt, 
        global_ep=global_ep, 
        res_queue=res_queue, 
        env_fn=make_env, 
        num_agents=3, 
        fov_size=10, 
        a_size=5, 
        device="cpu"
    ) for i in range(NUM_THREADS)]
    
    [w.start() for w in workers]
    episodes = []
    rewards = []

    try:
        while True:
            r = res_queue.get()
            if r is None:
                break
            
            ep, ep_reward, mode, bc_loss, steps, obs_dens = r
            mean_reward = np.mean(rewards[-100:]) if len(rewards) >= 1 else ep_reward

            print(
                f"[ep {ep:7d}] "
                f"reward={ep_reward:7.2f} "
                f"mean100={mean_reward:7.2f} "
                f"steps={steps:4d} "  
                f"mode={mode} "
                f"bc_loss={bc_loss} "
                f"Obstacle_density={obs_dens}"
            )
            episodes.append(ep)
            rewards.append(ep_reward)

            if ep % 10000 == 0:
                print(f">>> Automatic save after {ep} episodes...")
                torch.save(
                    {
                        "model": gnet.state_dict(),
                        "optimizer": opt.state_dict(),
                        "episode": ep
                    },
                    "checkpoint.pt"
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
                "checkpoint.pt"
            )
            print(">>> Emergency checkpoint saved successfully.")

            
            for w in workers:
                w.terminate()

    finally:
            [w.join() for w in workers]
            
            if len(episodes) > 0:
                plt.plot(episodes, rewards)
                plt.show()