from bigraph_schema.contract import ProcessContract

from v2ecoli.library.ecoli_step import EcoliProcess as Process


class GlobalClock(Process):
    """
    Track global time for Steps that do not rely on vivarium-core's built-in
    time stepping (see :ref:`timesteps`).
    """

    name = "global_clock"

    config_schema = {}

    description = (
        "Global clock — the shared simulation time for manually time-stepped Steps.\n\n"
        "Each iteration it advances a single global_time counter by the smallest "
        "interval that reaches the soonest process update:\n"
        "  Δt = min_p (next_update_time[p] − t);   t ← t + Δt.\n"
        "Because the increment is exactly that minimum, no process's scheduled "
        "update is ever stepped over. Steps that opt out of vivarium-core's built-in "
        "time stepping read this global_time to decide when they are next due."
    )

    contract = ProcessContract(
        summary=(
            "Holds the single global simulation-time counter and advances it by the minimum "
            "interval to the next scheduled process update, so manually time-stepped Steps share "
            "one clock and none is ever stepped over."
        ),
        symbols={
            "t": "the current global simulation time shared by all manually time-stepped Steps (seconds)",
            "next_update_time[p]": "the global time at which manually time-stepped process p is next due to update (seconds)",
            "Δt": "the interval global_time is advanced by this iteration — the minimum remaining time to any process's next update (seconds)",
        },
        math=[
            r"\Delta t = \min_{p} \big(\text{next\_update\_time}[p] - t\big)",
            r"t \leftarrow t + \Delta t",
        ],
        inputs={
            "global_time": "Reads t, the current shared global time, as the baseline the next-update times are measured from.",
            "next_update_time": (
                "Reads the map of each manually time-stepped process's next-due global time; their "
                "minimum-minus-t sets the step size."
            ),
        },
        outputs={
            "global_time": "Writes the advanced t (incremented by Δt), the clock every manually time-stepped Step reads.",
        },
        config={},
    )


    def inputs(self):
        return {
            'global_time': 'float',
            'next_update_time': 'map[float]',
        }

    def outputs(self):
        return {
            'global_time': 'float',
        }


    def calculate_timestep(self, interval_or_states, states=None):
        """Calculate the minimum time until a manually time-stepped process
        needs to update.

        Bridges v1 signature ``(states)`` and v2 signature ``(interval, state)``.
        """
        if states is None:
            # v1 call: calculate_timestep(states)
            view = interval_or_states
        else:
            # v2 call: calculate_timestep(interval, state)
            view = states
        return min(
            next_update_time - view["global_time"]
            for next_update_time in view["next_update_time"].values()
        )

    def update(self, states, interval):
        """
        The interval that we increment global_time by is the same minimum time step
        that we calculated in calculate_timestep. This guarantees that we never
        accidentally skip over a process update time.
        """
        return {"global_time": interval}
