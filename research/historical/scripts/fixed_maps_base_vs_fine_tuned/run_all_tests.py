import os
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


SCRIPTS_TO_RUN = [
    "maze/evaluate_agents.py",
    "room/evaluate_agents.py",  
    "warehouse/evaluate_agents.py"
]


MODES_TO_RUN = ["classic", "improved"]

def main():
    print("==================================================")
    print("🚀 STARTING SEQUENTIAL MAPF TEST SUITE 🚀")
    print("==================================================")

    
    for mode in MODES_TO_RUN:
        print(f"\n📢 {'*'*20} STARTING MODE: {mode.upper()} {'*'*20}")
        
        
        for script_rel_path in SCRIPTS_TO_RUN:
            script_full_path = os.path.join(BASE_DIR, script_rel_path)
            script_dir = os.path.dirname(script_full_path)
            script_name = os.path.basename(script_full_path)

            if not os.path.exists(script_full_path):
                print(f"\n[WARNING] File not found: {script_full_path}. Skipping.")
                continue

            print(f"\n{'-'*50}")
            print(f"▶ RUNNING: {script_rel_path} | Mode: {mode.upper()}")
            print(f"{'-'*50}\n")

            try:
                
                subprocess.run(
                    [sys.executable, script_name, mode],
                    cwd=script_dir,
                    check=True
                )
                print(f"\n✅ COMPLETED SUCCESSFULLY: {script_rel_path} [{mode.upper()}]")

            except subprocess.CalledProcessError as e:
                print(f"\n❌ [ERROR] Test {script_rel_path} in mode {mode.upper()} failed (code: {e.returncode}).")
                print("Moving to the next task...")
            
            except KeyboardInterrupt:
                print("\n🛑 [INTERRUPTED] User interrupted the tests (Ctrl+C).")
                return

    print("\n==================================================")
    print("🏁 ALL TESTS (CLASSIC AND IMPROVED) COMPLETED 🏁")
    print("==================================================")

if __name__ == "__main__":
    main()