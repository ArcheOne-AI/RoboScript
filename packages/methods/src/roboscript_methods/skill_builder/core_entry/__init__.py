"""The core entry check, the same for every benchmark: the core's manipulation API, with a
benchmark's bindings, end to end through its env planes in each task's demonstration scene,
every call judged by the simulator's truth (P1 builds skills, it does not execute as the Agent,
so it may read truth).

    move_to_pose  per arm, `POSES` poses the planner plans to (a top-down TCP at a probed
                  workspace cell over the table, clear of every object): the TCP's error
    displacement  per arm, `move_by` along z, x, y and back; then both arms at once
                  (`together`): the TCP's error against the commanded displacement
    gripper       per arm close and open, then both at once: the reading against the command
    grasp         per graspable object (a movable rigid body the open gripper spans, at most a
                  few), the arm the toolset's geometric rule gives (the nearer TCP; the other
                  where only it plans the grasp), `grasp_object` with the default spec and a lift:
                  held when truth has the object 5 cm higher
    place         the held object back where it was picked (`place_object`, a robot-frame
                  target): placed when truth has it resting within 3 cm of the target
    return_home  both arms at the end of each episode: within the tolerance the benchmark's
                    glue passes (`Bench.origin_tolerance`, its official value)

A receipt is honest unless it reports success where truth says otherwise (for a move: its TCP
ends farther than the move tolerance, what ok promises). A call is judged by `Criteria`, each
measured on a benchmark's runs by the gap rule (`gaps`) where its values leave a clear gap. A
benchmark's glue (an `entry_check.py`) starts its env server, chooses the scene (with
the objects its demonstration picks up) and calls `check.run_check`; `report` reads the reports
and judges the pool by the benchmark's gate file. The modules are imported one by one: `gaps` is
numpy only and runs in a simulator's interpreter.
"""
