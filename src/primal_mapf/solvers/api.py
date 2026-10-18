"""Common solver format: path[t][agent] = (row, column)."""
import importlib
import multiprocessing as mp
import numpy as np


def _native_solve(connection, name, blocked, starts, goals, timeout):
    try:
        module = importlib.import_module(f"primal_mapf.solvers.{name}_core")
        connection.send((True, module.find_path(blocked, starts, goals, float(timeout))))
    except Exception as exc:
        connection.send((False, repr(exc)))
    finally:
        connection.close()


def valid_plan(path, obstacles, starts, goals):
    """Return False for malformed plans as well as illegal MAPF transitions."""
    try:
        obstacles = np.asarray(obstacles)
        if obstacles.ndim != 2 or path is None or len(path) == 0:
            return False
        steps = []
        for step in path:
            normalized = []
            for point in step:
                if len(point) != 2 or any(not isinstance(v, (int, np.integer)) for v in point):
                    return False
                normalized.append(tuple(map(int, point)))
            steps.append(normalized)
        if steps[0] != list(map(tuple, starts)) or steps[-1] != list(map(tuple, goals)):
            return False
        previous = None
        for step in steps:
            if len(step) != len(starts) or len(set(step)) != len(step):
                return False
            for i, (r, c) in enumerate(step):
                if not (0 <= r < obstacles.shape[0] and 0 <= c < obstacles.shape[1]) or obstacles[r, c]:
                    return False
                if previous is not None:
                    if abs(r-previous[i][0]) + abs(c-previous[i][1]) > 1:
                        return False
                    for j in range(i):
                        if step[i] == previous[j] and step[j] == previous[i]:
                            return False
            previous = step
        return True
    except (TypeError, ValueError, IndexError):
        return False


def find_path(name, obstacles, starts, goals, max_steps=256, timeout=10, seed=42):
    obstacles = np.asarray(obstacles)
    if obstacles.ndim != 2 or not np.isin(obstacles, [0, 1]).all():
        raise ValueError("Obstacle map must be a two-dimensional binary array")
    if len(starts) == 0 or len(starts) != len(goals):
        raise ValueError("Supply equally sized non-empty starts and goals")
    for endpoints in (starts, goals):
        try:
            normalized = list(map(tuple, endpoints))
            if len(set(normalized)) != len(endpoints):
                raise ValueError("Duplicate endpoints")
            for point in normalized:
                if len(point) != 2 or any(not isinstance(v, (int, np.integer)) for v in point):
                    raise ValueError("Endpoints must be integer (row, column) pairs")
                r, c = point
                if not (0 <= r < obstacles.shape[0] and 0 <= c < obstacles.shape[1]) or obstacles[r, c]:
                    raise ValueError("Endpoint is outside free space")
        except TypeError as exc:
            raise ValueError("Endpoints must be integer (row, column) pairs") from exc
    starts, goals = list(map(tuple, starts)), list(map(tuple, goals))
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if name == "pibt":
        from .pypibt.pibt import PIBT
        path = PIBT(np.asarray(obstacles) == 0, starts, goals, seed=seed).run(max_timestep=max_steps)
    elif name in {"ca", "cbs"}:
        try:
            importlib.import_module(f"primal_mapf.solvers.{name}_core")
        except ImportError as exc:
            raise RuntimeError("Build native solvers with scripts/build_solvers.py first") from exc
        # Bindings infer dimensions from supplied coordinates. A blocked border
        # preserves the full map extent, including empty final rows/columns.
        padded = np.pad(obstacles, 1, constant_values=1)
        blocked = [tuple(map(int, p)) for p in np.argwhere(padded)]
        shifted_starts = [(int(r)+1, int(c)+1) for r, c in starts]
        shifted_goals = [(int(r)+1, int(c)+1) for r, c in goals]
        context = mp.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_native_solve,
                                  args=(sender, name, blocked, shifted_starts, shifted_goals, timeout))
        process.start()
        sender.close()
        try:
            if not receiver.poll(timeout):
                return None
            try:
                ok, paths = receiver.recv()
            except EOFError as exc:
                raise RuntimeError(f"Native solver {name} terminated unexpectedly") from exc
            if not ok:
                raise RuntimeError(paths)
        finally:
            receiver.close()
            process.join(timeout=0.1)
            if process.is_alive():
                process.terminate()
                process.join()
        if not paths or any(not p for p in paths):
            return None
        path = [[tuple(int(v)-1 for v in p[min(t, len(p)-1)]) for p in paths]
                for t in range(max(map(len, paths)))]
    else:
        raise ValueError(f"Unknown solver: {name}")
    return path if path and len(path)-1 <= max_steps and valid_plan(path, obstacles, starts, goals) else None
