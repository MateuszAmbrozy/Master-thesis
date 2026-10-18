import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
RANDOM_ROOT = os.path.dirname(current_file_path)
sys.path.append(RANDOM_ROOT)

import eval_core

if __name__ == "__main__":        
    custom_map_path = "room-64-64-16.map"
    
    map_name = os.path.splitext(os.path.basename(custom_map_path))[0]
    generated_test_name = f"agent_count_{map_name}"

    eval_core.run_evaluation(
        test_name=generated_test_name,
        param_name="agents",
        param_values=[4, 8, 16, 32],
        custom_map_path=custom_map_path,
        num_episodes=20
    )