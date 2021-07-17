import manim as M
import pandas as pd
import numpy as np


class Plot(M.Scene):
    def __init__(self):
        # Path of the files
        signal_path = "data/signal.csv"
        features_path = "data/features.csv"

        # Loading all the required data
        self.signal_frame = pd.read_csv(signal_path)
        self.features_frame = pd.read_csv(features_path)
        self.times = self.signal_frame["time"].values
        self.lead1 = self.signal_frame["lead1"].values
        self.lead2 = self.signal_frame["lead2"].values
        self.beat_idx = self.features_frame["position"].values

        # NOTE: Change the constants below as required

        # Time in seconds to create one cycle
        self.CYCLE_CREATION_RUNTIME = 1
        # A float between 0 and 1 determining the delay in
        # creation of a cycle and removal of previous cycle
        self.DELAY_BETWEEN_CYCLES = 0.1

        super().__init__()

    def setup_axes(self):
        """[Add the required axes to the scene]"""

        def get_grid(
            axes,
            color=M.WHITE,
            stroke_width=0.2,
        ):
            """[Given an axes it returns a lines of a grid]

            Returns:
                [VDict]: [Return a VDict containing the grid lines]
            """
            vertical_lines = M.VGroup()
            horizontal_lines = M.VGroup()

            x_start, x_end, x_step = axes.x_range
            y_start, y_end, y_step = axes.y_range

            for x in np.arange(x_start + x_step, x_end, x_step):
                start_point = axes.coords_to_point(x, y_start)
                end_point = axes.coords_to_point(x, y_end)
                line = M.Line(start_point, end_point).set_stroke(
                    color=color, width=stroke_width
                )
                vertical_lines.add(line)

            for y in np.arange(y_start + y_step, y_end, y_step):
                start_point = axes.coords_to_point(x_start, y)
                end_point = axes.coords_to_point(x_end, y)
                line = M.Line(start_point, end_point).set_stroke(
                    color=color, width=stroke_width
                )
                horizontal_lines.add(line)

            mappings = [
                ("vertical_lines", vertical_lines),
                ("horizontal_lines", horizontal_lines),
            ]
            grid = M.VDict(mappings)

            return grid

        # Axes for data of lead1
        self.lead1_axes = M.Axes(
            x_range=[0, 10, 1],
            y_range=[-0.2, 0.5, 0.1],
            x_length=6,
            y_length=3,
            axis_config={"include_tip": False, "number_scale_value": 0.3},
            x_axis_config={
                "numbers_to_include": np.arange(0, 10 + 1, 1),
            },
            y_axis_config={
                "decimal_number_config": {"num_decimal_places": 1},
                "numbers_to_include": np.arange(-0.2, 0.51, 0.1),
                "numbers_to_exclude": [],
            },
            tips=False,
        ).to_edge(M.UL)

        # Axes for data of lead2
        self.lead2_axes = M.Axes(
            x_range=[0, 10, 1],
            y_range=[-0.5, 0.2, 0.1],
            x_length=6,
            y_length=3,
            axis_config={"include_tip": False, "number_scale_value": 0.3},
            x_axis_config={
                "numbers_to_include": np.arange(0, 10 + 1, 1),
            },
            y_axis_config={
                "decimal_number_config": {"num_decimal_places": 1},
                "numbers_to_include": np.arange(-0.5, 0.21, 0.1),
                "numbers_to_exclude": [],
            },
            tips=False,
        ).to_edge(M.DL)

        # Adding the axes to the scene
        self.add(self.lead1_axes)
        self.add(get_grid(self.lead1_axes))
        self.add(self.lead2_axes)
        self.add(get_grid(self.lead2_axes))

    def setup_points(self):
        """[Sets up all the required data points for the scene]"""
        lead1_axes_x_max = self.lead1_axes.x_range[1]
        self.lead1_all_points = [
            self.lead1_axes.coords_to_point(
                self.times[i] % (lead1_axes_x_max), self.lead1[i]
            )
            for i in range(len(self.times))
        ]

        lead2_axes_x_max = self.lead2_axes.x_range[1]
        self.lead2_all_points = [
            self.lead2_axes.coords_to_point(
                self.times[i] % (lead2_axes_x_max), self.lead2[i]
            )
            for i in range(len(self.times))
        ]

        self.beats = [
            M.Dot(
                self.lead1_axes.coords_to_point(
                    self.times[i] % (lead2_axes_x_max), 0.5
                )
            )
            if i in self.beat_idx
            else None
            for i in range(len(self.times))
        ]

        # An array containing indexes where the graph should wrap around
        # along with the start and end index
        self.checkpoints = [0]
        num_lines = 0
        for i in range(1, len(self.times)):
            if self.times[i] % (lead2_axes_x_max) != 0:
                num_lines += 1
            else:
                self.checkpoints.append(i - 1)
        self.checkpoints = self.checkpoints + [num_lines]

    def construct(self):
        self.setup_axes()
        self.setup_points()

        # Keeping storage of the cycle for removal from scene later on
        # Used in the inner function (lines_with_beats)
        lead1_cycle = None
        lead2_cycle = None
        beats = None

        def lines_with_beats(start, end, line_color, create=True):
            """[Returns the animations for the graph and the beats]

            Args:
                start ([int]): [index of the checkpoint array to start at]
                end ([int]): [index of the checkpoint array to end at]
                line_color ([string]): [color of the graph]
                create (bool, optional):
                [boolean to determine if graph should be created or removed].
                Defaults to True.

            Returns:
                [list]: [contains the animations to be played out for a cycle]
            """
            animate_func = M.Create if create else M.Uncreate

            if create:
                start_idx = self.checkpoints[start] + 1
                end_idx = self.checkpoints[end]

                lead1_cycle_points = self.lead1_all_points[start_idx:end_idx]
                lead2_cycle_points = self.lead2_all_points[start_idx:end_idx]

                nonlocal lead1_cycle
                nonlocal lead2_cycle
                nonlocal beats

                lead1_cycle = M.VGroup().set_points_smoothly(
                    lead1_cycle_points
                )
                lead2_cycle = M.VGroup().set_points_smoothly(
                    lead2_cycle_points
                )
                beats = M.VGroup()

                for beat in self.beats[start_idx:end_idx]:
                    if beat:
                        beats.add(beat)
            else:
                lead1_cycle.reverse_points()
                lead2_cycle.reverse_points()

            animations = [
                animate_func(lead1_cycle.set_color(line_color)),
                animate_func(lead2_cycle.set_color(line_color)),
                animate_func(beats.set_color(M.RED)),
            ]

            return animations

        # Animating the creation of the graphs and beats
        for i in range(len(self.checkpoints) - 1):
            # Create the first cycle
            if i == 0:
                self.play(
                    M.AnimationGroup(
                        *lines_with_beats(i, i + 1, M.BLUE),
                        run_time=self.CYCLE_CREATION_RUNTIME,
                        rate_func=M.rate_functions.ease_in_out_expo,
                    )
                )
            # Creating the next cycle and removing the previous
            else:
                self.play(
                    M.LaggedStart(
                        M.AnimationGroup(
                            *lines_with_beats(i - 1, i, M.TEAL, create=False),
                        ),
                        M.AnimationGroup(
                            *lines_with_beats(i, i + 1, M.BLUE),
                        ),
                        lag_ratio=self.DELAY_BETWEEN_CYCLES,
                        run_time=2 * self.CYCLE_CREATION_RUNTIME,
                        rate_func=M.rate_functions.ease_in_out_expo,
                    )
                )

        self.wait()
