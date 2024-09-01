import typing as ty
from pathlib import Path
from warnings import warn

import ipywidgets as ipyw
import matplotlib.pyplot as plt
import networkx as nx
import traitlets as trt
from ipylab import Panel
from IPython.display import HTML, IFrame, display
from ipywidgets.widgets.trait_types import TypedTuple

import openmdao.api as om
import pymbe.api as pm

from .generator import make_circuit_interpretations

__all__ = (
    "CircuitUI",
    "draw_circuit",
    "make_circuit_interpretations",
    "update_multiplicities",
)

# The valid graph layouts available in networkx
NX_LAYOUTS = sorted(
    [
        nx_graph_layout.replace("_layout", "")
        for nx_graph_layout in dir(nx.layout)
        if nx_graph_layout.endswith("_layout")
        and not any(bad_stuff in nx_graph_layout for bad_stuff in ("partite", "rescale", "planar"))
    ]
)

DISCRETE_COLOR_SCALE = (
    "#0077bb",
    "#33bbee",
    "#009988",
    "#ee7733",
    "#cc3311",
    "#ee3377",
    "#bbbbbb",
)

NODE_COLORS = {
    "R": "#8888DD",
    "D": "#A020F0",
    "EMF": "#DD8888",
}


def update_multiplicities(
    circuit: pm.Model,
    num_resistors: ty.Tuple[int, int] = None,
    num_diodes: ty.Tuple[int, int] = None,
    num_connectors: ty.Tuple[int, int] = None,
) -> pm.Model:
    num_resistors = tuple(sorted(num_resistors or [])) or (3, 8)
    num_diodes = tuple(sorted(num_diodes or [])) or (1, 4)
    min_connectors = num_resistors[1] + num_diodes[1]
    num_connectors = tuple(sorted(num_connectors or [])) or (min_connectors, 2 * min_connectors)

    circuit_def = circuit.ownedElement["Circuit Builder"].ownedElement["Circuit"]
    multiplicity = circuit_def.ownedMember["Circuit Resistor"].multiplicity
    multiplicity.lowerBound._data["value"], multiplicity.upperBound._data["value"] = num_resistors

    multiplicity = circuit_def.ownedMember["Circuit Diode"].multiplicity
    multiplicity.lowerBound._data["value"], multiplicity.upperBound._data["value"] = num_diodes

    # Get the ConnectionUsage
    for member in circuit_def.ownedMember:
        if member._metatype != "ConnectionUsage":
            continue
        (
            member.multiplicity.lowerBound._data["value"],
            member.multiplicity.upperBound._data["value"],
        ) = num_connectors
    return circuit


def draw_circuit(
    circuit_graph: nx.DiGraph,
    figsize=None,
    rad=0.1,
    arrowsize=40,
    linewidth=2,
    layout="kamada_kawai",
):
    figsize = figsize or (20, 20)

    layout_algorithm = getattr(nx.layout, f"{layout}_layout")
    node_pos = layout_algorithm(circuit_graph)

    internal_edges = {(n1, n2) for n1, n2 in circuit_graph.edges if n1[0] == n2[0]}

    nodes = {node for node, _ in sum(map(list, circuit_graph.edges), [])}

    emf_nodes = [node for node in nodes if node.startswith("EMF")]
    if not emf_nodes:
        raise ValueError("No EMF node in graph!")
    if len(emf_nodes) > 1:
        warn(f"Found multiple EMF nodes ({emf_nodes}), will use {emf_nodes[0]}!")
    emf = emf_nodes[0]
    first_path, *other_paths = list(
        nx.all_simple_edge_paths(circuit_graph, (emf, "Pos"), (emf, "Neg"))
    )

    loop_edge_list = set(first_path).difference(internal_edges)
    encountered_edges = loop_edge_list | internal_edges
    for edges in other_paths:
        new_edges = set(edges).difference(encountered_edges)
        loop_edge_list |= new_edges
        encountered_edges |= new_edges

    _ = plt.figure(figsize=figsize)
    for node_name_designator, color in NODE_COLORS.items():
        poses = {
            node: loc for node, loc in node_pos.items() if node[0].startswith(node_name_designator)
        }
        nx.draw_networkx_nodes(
            circuit_graph, poses, nodelist=list(poses.keys()), node_size=1200, node_color=color
        )

    num_flow_paths = len(other_paths) + 1
    num_scale_colors = len(DISCRETE_COLOR_SCALE)

    edge_colors = [
        DISCRETE_COLOR_SCALE[idx % num_scale_colors] for idx in range(num_flow_paths)
    ] + ["gray"]
    styles = ["-"] * num_flow_paths + [(0, (5, 5))]
    arrowstyles = ["-|>"] * num_flow_paths + ["->"]

    for edgelist, style, edge_color, arrowstyle in zip(
        [[*loop_edge_list]] + [list(internal_edges)], styles, edge_colors, arrowstyles
    ):
        nx.draw_networkx_edges(
            circuit_graph,
            node_pos,
            arrowstyle=arrowstyle,
            edgelist=edgelist,
            edge_color=edge_color,
            width=linewidth,
            style=style,
            arrowsize=arrowsize,
            connectionstyle=f"arc3,rad={rad}",
        )

    labels = {
        (node, polarity): f"{node}{'-' if polarity=='Neg' else '+'}" for node, polarity in node_pos
    }
    label_options = {"boxstyle": "circle", "ec": "white", "fc": "white", "alpha": 0.0}

    _ = nx.draw_networkx_labels(
        circuit_graph,
        node_pos,
        labels,
        font_size=8,
        font_color="white",
        font_weight="bold",
        bbox=label_options,
    )
    plt.show()


class CircuitPlotter(ipyw.VBox):
    """A widget to plot circuits."""

    graph: nx.DiGraph = trt.Instance(nx.DiGraph, args=())
    edge_curvature: ipyw.FloatSlider = trt.Instance(
        ipyw.FloatSlider,
        kw=dict(description="Edge Curvature", value=0.25, min=0, max=0.5, step=0.05),
    )
    line_width: ipyw.FloatSlider = trt.Instance(
        ipyw.FloatSlider, kw=dict(description="Line Width", value=2, min=1, max=3, step=0.2)
    )
    graph_layout: ipyw.Dropdown = trt.Instance(
        ipyw.Dropdown, kw=dict(options=NX_LAYOUTS, label="kamada_kawai")
    )
    plot: ipyw.Output = trt.Instance(ipyw.Output, args=())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for control in (self.edge_curvature, self.line_width, self.graph_layout):
            control.observe(self._update_plot, "value")

    @trt.validate("children")
    def _validate_children(self, proposal: trt.Bunch) -> tuple:
        children = proposal.value
        if children:
            return children
        return [
            ipyw.VBox(
                [
                    self.edge_curvature,
                    self.line_width,
                    self.graph_layout,
                ]
            ),
            self.plot,
        ]

    @trt.observe("graph")
    def _update_plot(self, *_):
        if len(self.graph.nodes) < 1:
            return
        self.plot.clear_output()
        with self.plot:
            plt.clf()
            draw_circuit(
                self.graph,
                figsize=(20, 10),
                rad=self.edge_curvature.value,
                linewidth=self.line_width.value,
                layout=self.graph_layout.value,
            )


class CircuitGenerator(ipyw.VBox):
    """An M0 instance generator for circuits."""

    selector: ipyw.IntSlider = trt.Instance(
        ipyw.IntSlider,
        kw=dict(
            description="Interpretation",
            description_tooltip="Index of M0 interpretation selected",
            layout=dict(visibility="hidden", height="0px", width="auto"),
            min=1,
            max=1,
        ),
    )

    make_new: ipyw.Button = trt.Instance(
        ipyw.Button, kw=dict(icon="plus", tooltip="Make new instances", layout=dict(width="40px"))
    )
    max_tries: ipyw.IntSlider = trt.Instance(
        ipyw.IntSlider,
        kw=dict(
            description="Max attempts",
            min=2,
            value=5,
            max=20,
            layout=dict(width="100%", min_width="300px"),
        ),
    )

    num_resistors: ipyw.IntRangeSlider = trt.Instance(
        ipyw.IntRangeSlider,
        kw=dict(
            description="# Resistors", min=1, max=12, layout=dict(width="auto", min_width="300px")
        ),
    )
    num_diodes: ipyw.IntRangeSlider = trt.Instance(
        ipyw.IntRangeSlider,
        kw=dict(
            description="# Diodes", min=0, max=5, layout=dict(width="auto", min_width="300px")
        ),
    )

    progress_bar: ipyw.IntProgress = trt.Instance(
        ipyw.IntProgress,
        kw=dict(
            min=0,
            max=5,
            layout=dict(visibility="hidden", height="0px", width="auto", min_width="300px"),
        ),
    )

    circuit_model: pm.Model = trt.Instance(pm.Model)
    interpretations: tuple = trt.Tuple()
    selected_interpretation: dict = trt.Dict()

    @trt.default("circuit_model")
    def _make_circuit_model(self) -> pm.Model:
        circuit_file = Path(pm.__file__).parent / "../../tests/fixtures/Circuit Builder.json"
        model = pm.Model.load_from_file(circuit_file)
        model.max_multiplicity = 100
        return model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = [
            self.selector,
            ipyw.Label("Generate M0 interpretations:"),
            ipyw.VBox((self.num_resistors, self.num_diodes), layout=dict(width="100%")),
            ipyw.HBox((self.max_tries, self.make_new), layout=dict(width="100%")),
            self.progress_bar,
        ]
        self.selector.observe(self._update_interpretation, "value")
        for slider in (self.num_resistors, self.num_diodes):
            slider.observe(self.update_multiplicities, "value")

        self._update_on_new_interpretations()
        self.make_new.on_click(self._make_new_interpretations)

    @trt.observe("interpretations")
    def _update_on_new_interpretations(self, *_):
        interpretations = self.interpretations
        selector = self.selector
        if interpretations:
            selector.disabled = False
            selector.layout.visibility = "visible"
            selector.layout.height = "40px"
            selector.max = len(interpretations)
            if not self.selected_interpretation:
                self._update_interpretation()
        else:
            selector.disabled = True
            selector.layout.visibility = "hidden"
            selector.layout.height = "0px"

    def _update_interpretation(self, *_):
        self.selected_interpretation = self.interpretations[self.selector.value - 1]

    def _make_new_interpretations(self, *_):
        self.update_multiplicities()
        max_tries = self.max_tries.value
        progress_bar = self.progress_bar
        progress_bar.value, progress_bar.max = 0, max_tries
        bar_layout = progress_bar.layout
        bar_layout.visibility = "visible"
        bar_layout.height = "40px"

        def on_attempt():
            progress_bar.value += 1

        new = make_circuit_interpretations(
            m1_model=self.circuit_model,
            required_interpretations=1,
            max_tries=max_tries,
            on_attempt=on_attempt,
        )
        if not new:
            warn(f"All {max_tries} attempts to make new M0 interpretations failed!")
        bar_layout.visibility = "hidden"
        bar_layout.height = "0px"
        self.interpretations += new

    def update_multiplicities(self, *_):
        update_multiplicities(
            circuit=self.circuit_model,
            num_resistors=self.num_resistors.value,
            num_diodes=self.num_diodes.value,
        )


class ParametricExecutor(Panel):
    """
        A controller for a model's parameters, runs it, and displays the results.

    .. note::
        Currently only looks at the options in a model.

    """

    description: str = trt.Unicode("Parametric Model").tag(sync=True)
    float_sliders: ty.Tuple[ipyw.FloatSlider] = TypedTuple(trt.Instance(ipyw.FloatSlider))
    result_labels: ty.Tuple[ipyw.Label] = TypedTuple(trt.Instance(ipyw.Label))

    problem: om.Problem = trt.Instance(om.Problem, allow_none=True)
    inputs: ipyw.VBox = trt.Instance(ipyw.VBox)
    input_sliders: ipyw.VBox = trt.Instance(ipyw.VBox, args=())
    results: ipyw.VBox = trt.Instance(ipyw.VBox)

    run_problem: ipyw.Button = trt.Instance(ipyw.Button)

    on_run_callbacks: ty.Tuple[ty.Callable] = TypedTuple(trt.Callable())

    DEFAULT_DOCK_LAYOUT = {
        "type": "split-area",
        "orientation": "vertical",
        "children": [
            {"type": "tab-area", "widgets": [0], "currentIndex": 0},
            {"type": "tab-area", "widgets": [1], "currentIndex": 0},
        ],
        "sizes": [0.5, 0.5],
    }

    def __init__(self, *args, **kwargs):
        kwargs["dock_layout"] = kwargs.get("dock_layout", self.DEFAULT_DOCK_LAYOUT)
        super().__init__(*args, **kwargs)

    @trt.validate("children")
    def _validate_children(self, proposal: trt.Bunch) -> tuple:
        children = proposal.value
        if children:
            return children
        return tuple(
            [
                self.inputs,
                self.results,
            ]
        )

    @trt.default("inputs")
    def _make_inputs(self) -> ipyw.VBox:
        box = ipyw.VBox(
            children=(self.run_problem, self.input_sliders),
            layout=dict(height="100%", width="100%"),
        )
        box.add_traits(description=trt.Unicode("Model Parameters").tag(sync=True))
        return box

    @trt.default("results")
    def _make_results(self) -> ipyw.VBox:
        box = ipyw.VBox(layout=dict(height="100%", width="100%"))
        box.add_traits(description=trt.Unicode("Model Results").tag(sync=True))
        return box

    @trt.default("run_problem")
    def _make_run_problem_button(self) -> ipyw.Button:
        btn = ipyw.Button(
            icon="play",
            tooltip="Run OpenMDAO problem",
            disabled=True,
            layout=dict(width="40px"),
        )
        btn.on_click(self._run_problem)
        return btn

    def _run_problem(self, *_):
        self.run_problem.disabled = True
        self.results.children = []
        self.update_parameter_values()
        self.problem.run_model()
        # TODO: update problem values
        self.update_results(self.problem)

    def update_parameter_values(self):
        problem = self.problem
        for slider in self.input_sliders.children:
            name = slider.description
            value = slider.value
            try:
                problem.set_val(name, value)
            # If not an attribute, try to update the option of a system
            except KeyError:
                item = problem.model
                *keys, attribute = name.split(".")
                for key in keys:
                    item = getattr(item, key)
                item.options[attribute] = value

    @trt.observe("problem")
    def _on_new_problem(self, *_):
        problem = self.problem
        if not isinstance(problem, om.Problem):
            self.run_problem.disabled = True
            return
        self.update_parameter_controllers(problem=problem)
        self.update_results(problem=problem)
        self.run_problem.disabled = False

    @staticmethod
    def get_kwargs(
        name,
        spec: dict,
        num_steps: int = 20,
        upper_mult: float = 2.0,
        lower_mult: float = 0.5,
    ) -> dict:
        value = spec.get("val")
        if spec.get("shape") == (1,):
            value = value[0]
        upper = spec.get("upper")
        if upper is None:
            if not value:
                upper = 100  # TODO: figure out a better way to do this with other widgets
            upper = value * upper_mult
        lower = spec.get("lower")
        if lower is None:
            lower = value * lower_mult
        return dict(
            value=value,
            max=upper,
            min=lower,
            step=(upper - lower) / num_steps,
            description=name,
            description_tooltip=spec.get("desc") or f"Set the value for '{name}'",
        )

    def update_parameter_controllers(self, problem: om.Problem):
        # Get all the model systems' options that have a float as a value
        system_parameters = {
            system.pathname: {
                key: spec
                for key, spec in system.options._dict.items()
                if isinstance(spec.get("val"), float)
            }
            for system in problem.model.system_iter()
            if not system.name.startswith("_")
        }
        parameters = {
            f"{system_name}.{param_name}": self.get_kwargs(
                f"{system_name}.{param_name}", param_spec
            )
            for system_name, parameters in system_parameters.items()
            for param_name, param_spec in parameters.items()
        }
        parameters.update(
            {
                spec["prom_name"]: self.get_kwargs(spec["prom_name"], spec)
                for _, spec in problem.model.list_inputs(prom_name=True, out_stream=False)
            }
        )
        parameters = dict(sorted(parameters.items()))

        # if necessary, create new sliders
        num_sliders = len(parameters)
        slider_layout = dict(min_width="100px", width="98%")
        additional_sliders_required = num_sliders - len(self.float_sliders)
        if additional_sliders_required > 0:
            new_sliders = tuple(
                ipyw.FloatSlider(layout=slider_layout) for _ in range(additional_sliders_required)
            )
            for slider in new_sliders:
                slider.observe(self._update_run_button, "value")
            self.float_sliders += new_sliders

        # configure sliders
        sliders = self.float_sliders[:num_sliders]
        for slider, (_, parameter_spec) in zip(sliders, sorted(parameters.items())):
            slider.attribute = parameter_spec
            with self.hold_trait_notifications():
                # FIXME: figure out a better way to handle min/max/value checks
                slider.value = 0.5 * (slider.max + slider.min)
                slider.min = -9999999
                slider.max = 9999999
                for key, value in parameter_spec.items():
                    setattr(slider, key, value)

        self.input_sliders.children = sliders

    def update_results(self, problem: om.Problem):
        if not isinstance(problem, om.Problem):
            return
        outputs = {
            spec["prom_name"]: dict(
                value=spec["val"][0] if spec.get("shape") == (1,) else spec["val"],
                units=spec.get("units"),
                shape=spec.get("shape"),
            )
            for _, spec in problem.model.list_outputs(
                prom_name=True, shape=True, units=True, out_stream=False
            )
        }
        num_labels = len(outputs)
        additional_labels_required = num_labels - len(self.result_labels)
        if additional_labels_required > 0:
            self.result_labels += tuple(ipyw.Label() for _ in range(additional_labels_required))
        labels = self.result_labels[:num_labels]
        for label, (name, value) in zip(labels, sorted(outputs.items())):
            label.value = (
                f"{name}: {value['value']:.5g}" + f" {value['units']}" if value["units"] else ""
            )
        self.results.children = labels

    def _update_run_button(self, *_):
        self.run_problem.disabled = False
        for callback in self.on_run_callbacks:
            callback(problem=self.problem)


class CircuitUI(DockPop):
    """A user interface for interacting with the Circuit Builder example."""

    DEFAULT_LAYOUT = {
        "type": "split-area",
        "orientation": "horizontal",
        "children": [
            {
                "type": "split-area",
                "orientation": "vertical",
                "children": [
                    {"type": "tab-area", "widgets": [0], "currentIndex": 0},
                    {"type": "tab-area", "widgets": [4], "currentIndex": 0},
                ],
                "sizes": [0.28, 0.72],
            },
            {"type": "tab-area", "widgets": [1, 3, 2], "currentIndex": 0},
        ],
        "sizes": [0.2, 0.8],
    }

    panels: Panel = trt.Instance(
        Panel, kw=dict(description="Circuit Generator", dock_layout=DEFAULT_LAYOUT)
    )

    circuit_model: pm.Model = trt.Instance(pm.Model, allow_none=True)
    interpretation: dict = trt.Dict()

    instance_generator: CircuitGenerator = trt.Instance(CircuitGenerator, args=())
    graph_plotter: CircuitPlotter = trt.Instance(CircuitPlotter, args=())
    connections: ipyw.Output = trt.Instance(ipyw.Output, kw=dict(layout=dict(width="100%")))
    n2: ipyw.Output = trt.Instance(ipyw.Output, kw=dict(layout=dict(width="100%")))
    parametric_executor: ParametricExecutor = trt.Instance(
        ParametricExecutor, kw=dict(layout=dict(height="auto", width="100%"))
    )

    @trt.default("circuit_model")
    def _get_circuit_model(self) -> pm.Model:
        return self.instance_generator.circuit_model

    @property
    def panel_layout(self):
        return self.panels.dock_layout

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        generator = self.instance_generator

        if self.circuit_model is None:
            self.circuit_model = self.instance_generator.circuit_model

        panels = {
            "Generator": generator,
            "Graph": self.graph_plotter,
            "Connections": self.connections,
            "N2": self.n2,
            "Parametric Model": self.parametric_executor,
        }
        self.panels.children = tuple(panels.values())
        self.children = [self.panels]

        for name, widget in panels.items():
            widget.add_traits(description=trt.Unicode(default_value=name).tag(sync=True))

        self.parametric_executor.on_run_callbacks += (self._update_connections,)

        trt.link((self, "circuit_model"), (generator, "circuit_model"))
        trt.link((self, "interpretation"), (generator, "selected_interpretation"))

    @trt.observe("interpretation")
    def _update_graph(self, *_):
        graph = self.interpretation.get("cleaned_graph")
        if isinstance(graph, nx.DiGraph):
            self.graph_plotter.graph = graph
        problem = self.interpretation.get("problem")
        if isinstance(problem, om.Problem):
            self.parametric_executor.problem = problem

    def _update_n2(self, problem: om.Problem, filename="n2.html", width="100%", height=700):
        html_file = Path(filename).resolve().absolute()
        if html_file.exists():
            html_file.unlink()
        om.n2(problem, outfile=str(filename))
        self.n2.clear_output()
        if html_file.is_file():
            with self.n2:
                display(IFrame(str(filename), width=width, height=height))

    def _update_connections(self, problem: om.Problem, filename="connections.html"):
        html_file = Path(filename).resolve().absolute()
        if html_file.exists():
            html_file.unlink()
        om.view_connections(problem, outfile=str(filename))
        self.connections.clear_output()
        if html_file.is_file():
            with self.connections:
                display(HTML(html_file.read_text()))

    @trt.observe("interpretation")
    def _update_openmdao_views(self, change: trt.Bunch = None):
        if change:
            problem = (change.new or {}).get("problem")
        elif self.interpretation:
            problem = self.interpretation.get("problem")
        if problem:
            self._update_n2(problem=problem)
            self._update_connections(problem=problem)
