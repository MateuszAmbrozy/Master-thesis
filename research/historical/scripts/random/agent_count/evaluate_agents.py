import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
RANDOM_ROOT = os.path.dirname(current_file_path)
sys.path.append(RANDOM_ROOT)

import eval_core

if __name__ == "__main__":
    eval_core.run_evaluation(
        test_name="agent_count",
        param_name="agents",
        param_values=[4, 8, 16, 32],
        static_agents=None,
        static_size=(30, 30),
        static_density=0.1,
        num_episodes=20
    )