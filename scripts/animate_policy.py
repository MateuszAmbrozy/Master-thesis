"""Run a fine-tuned policy and export its actual trajectory as a GIF."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from primal_mapf.animation import render_animation
from primal_mapf.fixed_maps import MAPS
from primal_mapf.runtime import ROOT, load_model, project_path, scenario, seed_all
from primal_mapf.evaluation.evaluate import policy_episode
from primal_mapf.provenance import file_sha256, runtime_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--map", choices=MAPS)
    source.add_argument("--from-trace", help="Render a previously recorded trajectory without running inference")
    parser.add_argument("--variant", choices=["classic","improved"], default="improved")
    parser.add_argument("--checkpoint", help="Defaults to the selected map's fine-tuned checkpoint")
    parser.add_argument("--agents",type=int,default=8)
    parser.add_argument("--seed",type=int,default=123)
    parser.add_argument("--max-steps",type=int,default=512)
    parser.add_argument("--execution",choices=["sequential","joint"],default="sequential")
    parser.add_argument("--environment-profile",choices=["historical","common"],default="historical")
    parser.add_argument("--greedy",action="store_true")
    parser.add_argument("--fps",type=int,default=12)
    parser.add_argument("--max-frames",type=int,default=240)
    parser.add_argument("--width",type=int,default=1200)
    parser.add_argument("--mp4",action="store_true",help="Also export MP4 using ffmpeg")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    output = project_path(args.output)
    try:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Animation output must be empty; choose another --output directory")
        if not 1 <= args.fps <= 50 or args.width < 480 or args.max_frames < 2:
            raise ValueError("fps must be in 1..50, width >= 480, and max_frames >= 2")
        if args.mp4 and shutil.which("ffmpeg") is None:
            raise RuntimeError("MP4 export requires ffmpeg")
        if args.from_trace:
            record = json.loads(project_path(args.from_trace).read_text())
            local_checkpoint = project_path(record["checkpoint"])
            if local_checkpoint.exists() and file_sha256(local_checkpoint) != record["checkpoint_sha256"]:
                raise ValueError("Checkpoint hash differs from the recorded model")
        else:
            if args.agents < 1 or args.max_steps < 1 or args.seed < 0:
                raise ValueError("Agent/step limits must be positive and seed non-negative")
            torch.set_num_threads(1)
            seed_all(args.seed)
            checkpoint = args.checkpoint or f"checkpoints/fine_tuned/{args.variant}/{args.map}.pt"
            model = load_model(args.variant,checkpoint)
            grid,starts,goals = scenario(dict(map=MAPS[args.map],agents=args.agents),args.seed)
            begin = time.perf_counter()
            success,makespan,trace = policy_episode(model,args.variant,grid,starts,goals,args.max_steps,args.seed,
                                                    args.greedy,True,execution=args.execution,profile=args.environment_profile)
            record = dict(map_name=args.map,map=MAPS[args.map],variant=args.variant,checkpoint=checkpoint,
                          checkpoint_sha256=file_sha256(project_path(checkpoint)),seed=args.seed,agents=args.agents,
                          max_steps=args.max_steps,greedy=args.greedy,execution=args.execution,
                          environment_profile=args.environment_profile,success=success,makespan=makespan,
                          seconds=time.perf_counter()-begin,obstacles=grid.tolist(),starts=starts,goals=goals,trace=trace)
        details = render_animation(record,output,fps=args.fps,max_frames=args.max_frames,width=args.width,mp4=args.mp4)
        (output/"metadata.json").write_text(json.dumps(runtime_metadata(ROOT),indent=2)+"\n")
        print(f"success={record['success']} makespan={record['makespan']} agents={record['agents']}")
        print(f"Saved {output/'animation.gif'} ({details['gif_bytes']/1024:.0f} KiB)")
    except (ValueError,FileExistsError,FileNotFoundError,RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
