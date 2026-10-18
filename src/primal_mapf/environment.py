import numpy as np
import random
from dataclasses import dataclass
from typing import NamedTuple
from collections import deque
import sys
import heapq

opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
JOINT = False
dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
actionDict={v:k for k,v in dirDict.items()}

class Map():
    def __init__(self, goals, state, num_agents=1):
        self.state = state.copy()
        self.goals = goals.copy()
        self.num_agents = num_agents
        self.agents, self.agents_past, self.agent_goals = self.scanForAgents()
        assert(len(self.agents) == num_agents)

    def scanForAgents(self):
        agents = [(-1,-1) for i in range(self.num_agents)]
        agents_last = [(-1,-1) for i in range(self.num_agents)]
        agent_goals = [(-1,-1) for i in range(self.num_agents)]
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if(self.state[i,j]>0):
                    agents[self.state[i,j]-1] = (i,j)
                    agents_last[self.state[i,j]-1] = (i,j)
                if(self.goals[i,j]>0):
                    agent_goals[self.goals[i,j]-1] = (i,j)
        assert((-1,-1) not in agents and (-1,-1) not in agent_goals)
        assert(agents==agents_last)
        return agents, agents_last, agent_goals

    def getAgentPos(self, agent_id):
        return self.agents[agent_id-1]

    def getPastPos(self, agent_id):
        return self.agents_past[agent_id-1]

    def getAction(self,direction):
        return actionDict[direction]

    def getGoal(self, agent_id):
        return self.agent_goals[agent_id-1]

    def getObstacles(self):
        return (self.state==-1).astype(int)

    def moveAgent(self, direction, agent_id):
        ax=self.agents[agent_id-1][0]
        ay=self.agents[agent_id-1][1]

        # Not moving is always allowed
        if(direction==(0,0)):
            self.agents_past[agent_id-1]=self.agents[agent_id-1]
            return 1 if self.goals[ax,ay]==agent_id else 0

        # Otherwise, let's look at the validity of the move
        dx,dy = direction[0], direction[1]
        if(ax+dx>=self.state.shape[0] or ax+dx<0 or ay+dy>=self.state.shape[1] or ay+dy<0): #out of bounds
            return -1
        if(self.state[ax+dx,ay+dy]<0): #collide with static obstacle
            return -2
        if(self.state[ax+dx,ay+dy]>0): #collide with robot
            return -3

        # No collision: we can carry out the action
        self.state[ax,ay] = 0
        self.state[ax+dx,ay+dy] = agent_id
        self.agents_past[agent_id-1]=self.agents[agent_id-1]
        self.agents[agent_id-1] = (ax+dx,ay+dy)
        if self.goals[ax+dx,ay+dy]==agent_id:
            return 1
        elif self.goals[ax+dx,ay+dy]!=agent_id and self.goals[ax,ay]==agent_id:
            return 2
        else:
            return 0

    def act(self, action, agent_id):
        direction = dirDict[action]
        moved = self.moveAgent(direction, agent_id)
        return moved

    def getDir(self,action):
        return dirDict[action]

    def done(self):
        numComplete = 0
        for i in range(0, len(self.agents)):
            if self.goals[self.agents[i][0], self.agents[i][1]] == i+1:
                numComplete+=1
        return numComplete==len(self.agents)

@dataclass(frozen=True)
class EnvironmentSettings:
    """Reward and observation rules, independent of network architecture."""
    move: float = -0.3
    stay_off: float = -0.5
    stay_on: float = 0.0
    collision: float = -2.0
    blocking: float = -2.0
    blocking_inflation: int = 10
    distance_cap: float | None = None


PROFILES = {
    "common": EnvironmentSettings(),
    "classic": EnvironmentSettings(distance_cap=75.0),
    "improved": EnvironmentSettings(stay_off=-0.6, blocking=-1.5, blocking_inflation=6),
}


class StepResult(NamedTuple):
    observation: tuple
    reward: float
    done: bool
    valid_actions: list[int]
    on_goal: bool
    blocking: bool
    valid_action: bool


class JointStepResult(NamedTuple):
    rewards: list[float]
    done: bool
    blocked: list[bool]
    executed: list[bool]


class MAPFEnv:
    """Grid environment with explicit sequential and simultaneous step APIs.

    This is a project-specific interface. Gymnasium users should use
    ``primal_mapf.gym_env.JointMAPFEnv``.
    """
    metadata = {"render_modes": ["human"]}

    def __init__( self, SIZE=(10,40), num_agents=3, fov_size=10, obstacle_density=(0,.5), max_steps=256, settings=None):
        self.settings = settings or PROFILES["common"]
        self.individual_rewards = [0 for i in range(num_agents)]
        self.SIZE = SIZE
        self.num_agents = num_agents
        self.FOV_SIZE = fov_size
        self.OBSTACLE_DENSITY = obstacle_density
        self.max_steps = max_steps
        self.paths = None

        self.viewer = None
        self.finished = False
        self.really_obstacles_per = None

    def getObstacleMap(self):
        return self.world.getObstacles()

    def getGoals(self):
        result = []
        for i in range(1, self.num_agents + 1):
            result.append(self.world.getGoal(i))
        return result

    def getAgents(self):
        result = []
        for i in range(1, self.num_agents + 1):
            result.append(self.world.getAgentPos(i))
        return result

    def get_obs(self, agent_id):
        assert(agent_id>0)
        top_left=(self.world.getAgentPos(agent_id)[0]-self.FOV_SIZE//2,self.world.getAgentPos(agent_id)[1]-self.FOV_SIZE//2)
        bottom_right=(top_left[0]+self.FOV_SIZE,top_left[1]+self.FOV_SIZE)
        obs_shape=(self.FOV_SIZE,self.FOV_SIZE)
        goal_map             = np.zeros(obs_shape)
        poss_map             = np.zeros(obs_shape)
        goals_map            = np.zeros(obs_shape)
        obs_map              = np.zeros(obs_shape)
        visible_agents=[]
        for i in range(top_left[0],top_left[0]+self.FOV_SIZE):
            for j in range(top_left[1],top_left[1]+self.FOV_SIZE):
                if i>=self.world.state.shape[0] or i<0 or j>=self.world.state.shape[1] or j<0:
                    obs_map[i-top_left[0],j-top_left[1]]=1
                    continue
                if self.world.state[i,j]==-1:
                    obs_map[i-top_left[0],j-top_left[1]]=1
                if self.world.state[i,j]==agent_id:
                    poss_map[i-top_left[0],j-top_left[1]]=1
                if self.world.goals[i,j]==agent_id:
                    goal_map[i-top_left[0],j-top_left[1]]=1
                if self.world.state[i,j]>0 and self.world.state[i,j]!=agent_id:
                    visible_agents.append(self.world.state[i,j])
                    poss_map[i-top_left[0],j-top_left[1]]=1

        for agent in visible_agents:
            x, y = self.world.getGoal(agent)
            min_node = (max(top_left[0], min(top_left[0] + self.FOV_SIZE - 1, x)),
                        max(top_left[1], min(top_left[1] + self.FOV_SIZE - 1, y)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx=self.world.getGoal(agent_id)[0]-self.world.getAgentPos(agent_id)[0]
        dy=self.world.getGoal(agent_id)[1]-self.world.getAgentPos(agent_id)[1]
        mag=(dx**2+dy**2)**.5
        if self.settings.distance_cap is not None:
            mag = min(mag, self.settings.distance_cap)
        if mag!=0:
            dx=dx/mag
            dy=dy/mag
        return ([poss_map,goal_map,goals_map,obs_map],[dx,dy,mag])

    def reachable(self, world0):
        sys.setrecursionlimit(10000)
        world0 = world0.copy()

        def bfs(world0, start_x, start_y):
            visited = set()
            queue = deque([(start_x, start_y)])
            while queue:
                x, y = queue.popleft()
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < world0.shape[0] and 0 <= ny < world0.shape[1] and world0[nx, ny] == 0:
                        queue.append((nx, ny))
            return visited

        for i in range(world0.shape[0]):
            for j in range(world0.shape[1]):
                if world0[i, j] == 0:
                    visited = bfs(world0, i, j)
                    total_free_spaces = set(zip(*np.where(world0 == 0)))
                    return visited == total_free_spaces
        return False

    def astar(self, world, start, goal, robots=None):
        grid = world.copy()
        for (x, y) in robots or []:
            grid[x, y] = 1

        rows, cols = grid.shape
        def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dx, dy in neighbors:
                nx = current[0] + dx
                ny = current[1] + dy
                if nx < 0 or ny < 0 or nx >= rows or ny >= cols: continue
                if grid[nx, ny] == 1: continue

                neighbor = (nx, ny)
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return None

    def listNextValidActions(self, agent_id, prev_action=0, episode=0):
        available_actions = [0]
        agent_pos = self.world.getAgentPos(agent_id)
        ax,ay     = agent_pos[0],agent_pos[1]
        n_moves   = 5

        for action in range(1,n_moves):
            direction = self.world.getDir(action)
            dx,dy     = direction[0],direction[1]
            if(ax+dx>=self.world.state.shape[0] or ax+dx<0 or ay+dy>=self.world.state.shape[1] or ay+dy<0): continue
            if(self.world.state[ax+dx,ay+dy]<0): continue
            if(self.world.state[ax+dx,ay+dy]>0): continue
            available_actions.append(action)

        if opposite_actions[prev_action] in available_actions:
            available_actions.remove(opposite_actions[prev_action])

        return available_actions

    def get_blocking_reward(self,agent_id):
        other_robots=[]
        other_locations=[]
        inflation=self.settings.blocking_inflation
        top_left=(self.world.getAgentPos(agent_id)[0]-self.FOV_SIZE//2,self.world.getAgentPos(agent_id)[1]-self.FOV_SIZE//2)
        bottom_right=(top_left[0]+self.FOV_SIZE,top_left[1]+self.FOV_SIZE)
        for agent in range(1,self.num_agents + 1):
            if agent==agent_id: continue
            x,y=self.world.getAgentPos(agent)
            if x<top_left[0] or x>=bottom_right[0] or y>=bottom_right[1] or y<top_left[1]:
                continue
            other_robots.append(agent)
            other_locations.append((x,y))
        num_blocking=0
        world=self.getObstacleMap()
        for agent in other_robots:
            other_locations.remove(self.world.getAgentPos(agent))
            path_before=self.astar(world,self.world.getAgentPos(agent),self.world.getGoal(agent),
                                   robots=other_locations+[self.world.getAgentPos(agent_id)])
            path_after=self.astar(world,self.world.getAgentPos(agent),self.world.getGoal(agent),
                                   robots=other_locations)
            other_locations.append(self.world.getAgentPos(agent))
            if (path_before is None and path_after is None):continue
            if (path_before is not None and path_after is None):continue
            if (path_before is None and path_after is not None) or len(path_before)>len(path_after)+inflation:
                num_blocking+=1
        return num_blocking*self.settings.blocking

    def generate_map(self):
        def getConnectedRegion(world,regions_dict,x,y):
            sys.setrecursionlimit(1000000)
            if (x,y) in regions_dict: return regions_dict[(x,y)]
            visited=set()
            sx,sy=world.shape[0],world.shape[1]
            work_list=[(x,y)]
            while len(work_list)>0:
                (i,j)=work_list.pop()
                if(i<0 or i>=sx or j<0 or j>=sy): continue
                if(world[i,j]==-1): continue
                if world[i,j]>0: regions_dict[(i,j)]=visited
                if (i,j) in visited: continue
                visited.add((i,j))
                work_list.append((i+1,j))
                work_list.append((i,j+1))
                work_list.append((i-1,j))
                work_list.append((i,j-1))
            regions_dict[(x,y)]=visited
            return visited

        if self.OBSTACLE_DENSITY[0] == self.OBSTACLE_DENSITY[1]:
            prob = self.OBSTACLE_DENSITY[0]
        else:
            prob = np.random.triangular(
                self.OBSTACLE_DENSITY[0],
                0.33 * self.OBSTACLE_DENSITY[0] + 0.66 * self.OBSTACLE_DENSITY[1],
                self.OBSTACLE_DENSITY[1]
            )
            self.really_obstacles_per=prob

        size=np.random.choice([self.SIZE[0],self.SIZE[0]*.5+self.SIZE[1]*.5,self.SIZE[1]],p=[.5,.25,.25])
        world     = -(np.random.rand(int(size),int(size))<prob).astype(int)

        agent_counter = 1
        agent_locations=[]
        while agent_counter<=self.num_agents:
            x,y = np.random.randint(0,world.shape[0]),np.random.randint(0,world.shape[1])
            if(world[x,y] == 0):
                world[x,y]=agent_counter
                agent_locations.append((x,y))
                agent_counter += 1

        goals = np.zeros(world.shape).astype(int)
        goal_counter = 1
        agent_regions=dict()
        while goal_counter<=self.num_agents:
            agent_pos=agent_locations[goal_counter-1]
            valid_tiles=getConnectedRegion(world,agent_regions,agent_pos[0],agent_pos[1])
            x,y  = random.choice(list(valid_tiles))
            if(goals[x,y]==0 and world[x,y]!=-1):
                goals[x,y]    = goal_counter
                goal_counter += 1

        self.initial_world = world
        self.initial_goals = goals
        self.world = Map(goals=goals, state=world, num_agents=self.num_agents)

    def reset(self):
        self.finished = False
        self.generate_map()
        self.agent_positions = self.getAgents()
        self.agent_goals = self.getGoals()
        self.agent_reached = [False]*self.num_agents

        self.ACTIONS = dirDict
        self.GRID_SIZE = self.world.state.shape[0]
        self.OBSTACLES = self.getObstacleMap()

        observations = []
        for agent_id in range(1, self.num_agents + 1):
            observations.append(self.get_obs(agent_id))

        return observations

    def step(self, action_input,episode=0):
        self.fresh = False
        n_actions = 5

        if len(action_input) != 2 or action_input[1] not in range(n_actions) or action_input[0] not in range(1, self.num_agents+1):
            raise ValueError('Expected (agent_id, action) with agent_id in 1..N and action in 0..4')

        agent_id = action_input[0]
        action   = action_input[1]

        action_status = self.world.act(action,agent_id)
        valid_action = action_status >= 0
        blocking = False


        if action==0: #staying still
            if action_status == 1: #stayed on goal
                reward = self.settings.stay_on
                x = self.get_blocking_reward(agent_id)
                reward += x
                if x < 0:
                    blocking = True
            elif action_status == 0: #stayed off goal
                reward = self.settings.stay_off
        else: #moving
            if (action_status == 1): # reached goal
                reward = self.settings.stay_on
            elif (action_status == -3 or action_status==-2 or action_status==-1): # collision
                reward = self.settings.collision
            elif (action_status == 2): #left goal
                reward = self.settings.move
            else:
                reward = self.settings.move

        self.individual_rewards[agent_id-1] = reward

        if JOINT:
            visible=[False for i in range(self.num_agents)]
            v=0
            for agent in range(1,self.num_agents+1):
                if agent==agent_id: continue
                top_left=(self.world.getAgentPos(agent_id)[0]-self.FOV_SIZE//2, self.world.getAgentPos(agent_id)[1]-self.FOV_SIZE//2)
                pos=self.world.getAgentPos(agent)
                if pos[0]>=top_left[0] and pos[0]<top_left[0]+self.FOV_SIZE and pos[1]>=top_left[1] and pos[1]<top_left[1]+self.FOV_SIZE:
                    v+=1
                    visible[agent-1]=True
            if v>0:
                reward=self.individual_rewards[agent_id-1]/2
                for i in range(self.num_agents):
                    if visible[i]:
                        reward+=self.individual_rewards[i]/(v*2)

        state = self.get_obs(agent_id)
        done = self.world.done()
        self.finished |= done
        nextActions = self.listNextValidActions(agent_id, action, episode=episode)
        on_goal = self.world.getAgentPos(agent_id) == self.world.getGoal(agent_id)

        return StepResult(state, float(reward), done, nextActions, on_goal, blocking, valid_action)

    def set_custom_map(self, custom_obstacle_map, custom_starts, custom_goals):


        custom_obstacle_map = np.asarray(custom_obstacle_map)
        if custom_obstacle_map.ndim != 2 or not np.isin(custom_obstacle_map, [0, 1]).all():
            raise ValueError("Obstacle map must be a two-dimensional binary array")
        if len(custom_starts) != self.num_agents or len(custom_goals) != self.num_agents:
            raise ValueError("Starts and goals must match num_agents")
        for label, endpoints in [("starts", custom_starts), ("goals", custom_goals)]:
            if len(set(map(tuple, endpoints))) != self.num_agents:
                raise ValueError(f"Duplicate {label}")
            for point in endpoints:
                if len(point) != 2 or any(not isinstance(v, (int, np.integer)) for v in point):
                    raise ValueError(f"Invalid {label} coordinate")
                r, c = point
                if not (0 <= r < custom_obstacle_map.shape[0] and 0 <= c < custom_obstacle_map.shape[1]) or custom_obstacle_map[r, c]:
                    raise ValueError(f"Invalid {label}: endpoint outside free space")
        custom_starts = list(map(tuple, custom_starts))
        custom_goals = list(map(tuple, custom_goals))
        self.SIZE = custom_obstacle_map.shape
        self.OBSTACLE_DENSITY = (0.0, 0.0)


        state = np.zeros_like(custom_obstacle_map, dtype=int)
        state[custom_obstacle_map == 1] = -1

        goals = np.zeros_like(custom_obstacle_map, dtype=int)


        for i, start_pos in enumerate(custom_starts):
            agent_id = i + 1
            state[start_pos[0], start_pos[1]] = agent_id

        for i, goal_pos in enumerate(custom_goals):
            agent_id = i + 1
            goals[goal_pos[0], goal_pos[1]] = agent_id


        self.world = Map(goals=goals, state=state, num_agents=self.num_agents)
        self.initial_world = state.copy()
        self.initial_goals = goals.copy()


        self.finished = False
        self.agent_positions = self.getAgents()
        self.agent_goals = self.getGoals()
        self.agent_reached = [False] * self.num_agents

        self.ACTIONS = dirDict
        self.GRID_SIZE = self.world.state.shape[0]
        self.OBSTACLES = self.getObstacleMap()


        observations = []
        for agent_id in range(1, self.num_agents + 1):
            observations.append(self.get_obs(agent_id))

        return observations


    def legal_actions(self, agent_id: int) -> list[int]:
        """Static legal moves for a simultaneous timestep, including occupied cells.

        Inter-agent conflicts are resolved by step_joint, after all agents have
        selected actions from the same pre-transition state.
        """
        if agent_id not in range(1, self.num_agents + 1):
            raise ValueError("agent_id must be in 1..N")
        r, c = self.world.getAgentPos(agent_id)
        grid = self.world.state
        return [a for a in range(5)
                if 0 <= r + dirDict[a][0] < grid.shape[0]
                and 0 <= c + dirDict[a][1] < grid.shape[1]
                and grid[r + dirDict[a][0], c + dirDict[a][1]] >= 0]

    def step_joint(self, actions: list[int]) -> JointStepResult:
        """Apply one simultaneous transition with deterministic conflict handling.

        All vertex-conflicting movers and both sides of an edge swap stay.
        Cancellation propagates to movers targeting a newly stationary agent.
        Following moves and rotations of length >= 3 are allowed.
        """
        if len(actions) != self.num_agents or any(
                not isinstance(a, (int, np.integer)) or a not in range(5) for a in actions):
            raise ValueError("Supply one integer action in 0..4 per agent")
        before = self.getAgents()
        targets = []
        rejected = set()
        for i, (r, c) in enumerate(before):
            a = int(actions[i])
            if a not in self.legal_actions(i + 1):
                rejected.add(i)
                targets.append((r, c))
            else:
                dr, dc = dirDict[a]
                targets.append((r + dr, c + dc))
        # Conflicts can be introduced by cancelled moves; iterate to a fixed point.
        while True:
            cancel = set()
            owners = {}
            for i, target in enumerate(targets):
                owners.setdefault(target, []).append(i)
            for group in owners.values():
                if len(group) > 1:
                    cancel.update(i for i in group if targets[i] != before[i])
            for i in range(self.num_agents):
                for j in range(i):
                    if targets[i] == before[j] and targets[j] == before[i]:
                        cancel.update((i, j))
            if not cancel:
                break
            rejected.update(cancel)
            for i in cancel:
                targets[i] = before[i]
        state = np.where(self.world.state < 0, -1, 0)
        for i, (r, c) in enumerate(targets):
            state[r, c] = i + 1
        self.world.state = state
        self.world.agents_past = before
        self.world.agents = targets
        rewards, blocked = [], []
        for i, position in enumerate(targets):
            block = False
            if i in rejected:
                reward = self.settings.collision
            elif actions[i] == 0:
                if position == self.world.getGoal(i + 1):
                    penalty = self.get_blocking_reward(i + 1)
                    reward = self.settings.stay_on + penalty
                    block = penalty < 0
                else:
                    reward = self.settings.stay_off
            else:
                reward = self.settings.stay_on if position == self.world.getGoal(i + 1) else self.settings.move
            rewards.append(float(reward))
            blocked.append(block)
        self.individual_rewards = rewards
        done = self.world.done()
        self.finished |= done
        return JointStepResult(rewards, done, blocked,
                               [i not in rejected for i in range(self.num_agents)])
