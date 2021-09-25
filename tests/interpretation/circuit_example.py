import traceback
import typing as ty
from importlib import import_module
from pathlib import Path
from warnings import warn

import ipywidgets as ipyw
import matplotlib.pyplot as plt
import networkx as nx
import openmdao.api as om
import traitlets as trt
from IPython.display import HTML, IFrame, display
from wxyz.lab import DockBox, DockPop

import pymbe.api as pm
from pymbe.interpretation.interp_playbooks import random_generator_playbook
from pymbe.model import Element

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


def baseline_openmdao_example_circuit() -> dict:
    edges = (
        (("EMF", "Pos"), ("R1", "Neg")),
        (("EMF", "Pos"), ("R2", "Neg")),
        (("R1", "Pos"), ("EMF", "Neg")),
        (("R2", "Pos"), ("D1", "Neg")),
        (("D1", "Pos"), ("EMF", "Neg")),
    )
    baseline_digraph = nx.DiGraph()
    baseline_digraph.add_edges_from(edges)
    return dict(
        cleaned_graph=baseline_digraph,
        emf_pins={"+": ("EMF", "Pos"), "-": ("EMF", "Neg")},
        params=dict(R1=dict(R=100), R2=dict(R=10000)),
    )


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


def get_circuit_data(interpretation: dict, connector_feature: Element):
    """
    A very rudimentary filter for simple circuit connectors.

    Filters out self-connections and duplicate connectors between the same m0 pins.
    """
    unique_connections = get_unique_connections(interpretation, connector_feature)
    circuit_data = dict(interpretation=interpretation)

    def get_el_name(element):
        name = element.name
        name, num = name.split("#")
        name = name if name == "EMF" else name[0]
        return name + num

    def legal_predecessors(graph, node, valids):
        all_pred = set(graph.predecessors(node))
        return {pred for pred in all_pred if pred in valids}

    def legal_successors(graph, node, valids):
        all_suc = set(graph.successors(node))
        return {pred for pred in all_suc if pred in valids}

    graph = nx.DiGraph()
    edges = [
        ((get_el_name(source[-2]), "Pos"), (get_el_name(target[-2]), "Neg"))
        for source, target in unique_connections
    ]
    nodes = {node for node, _ in sum(map(list, edges), [])}
    edges += [((node, "Neg"), (node, "Pos")) for node in nodes if not node.startswith("EMF")]
    graph.add_edges_from(edges)

    emf_nodes = {node for node in nodes if node.upper().startswith("EMF")}
    if not emf_nodes:
        raise RuntimeError("There are not EMF nodes in the graph!")

    if len(emf_nodes) > 1:
        raise RuntimeError(
            f"Found {len(emf_nodes)} EMF nodes ({emf_nodes})!  There should only be one!"
        )
    emf_node = emf_nodes.pop()
    emf_pos_name = (emf_node, "Pos")
    emf_neg_name = (emf_node, "Neg")

    # find all paths in the graph from EMF positive side to EMF negative side

    flows = list(nx.all_simple_paths(graph, emf_pos_name, emf_neg_name))
    flow_edges = list(nx.all_simple_edge_paths(graph, emf_pos_name, emf_neg_name))

    flowed_nodes = {node for flow in flows for node in flow}

    circuit_data["valid_edges"] = set(sum(flow_edges, []))

    valid_graph = nx.DiGraph()
    valid_graph.add_edges_from(circuit_data["valid_edges"])

    # look at junctions in the graph

    # split junctions - where a positive pin has multiple outputs
    circuit_data["split_junctions"] = {
        node: legal_successors(valid_graph, node, flowed_nodes)
        for node in graph.nodes
        if len(legal_successors(valid_graph, node, flowed_nodes)) > 1
    }

    # join junctions - where a negative pin has multiple inputs
    circuit_data["join_junctions"] = {
        node: legal_predecessors(valid_graph, node, flowed_nodes)
        for node in graph.nodes
        if len(legal_predecessors(valid_graph, node, flowed_nodes)) > 1
    }

    circuit_data["both_ways_junctions"] = {
        node
        for node in graph.nodes
        if len(legal_successors(valid_graph, node, flowed_nodes)) > 1
        if node in sum(map(list, circuit_data["join_junctions"].values()), [])
    }

    circuit_data["number_loops"] = len(flows)
    circuit_data["valid_nodes"] = flowed_nodes

    circuit_data["loop_edges"] = flow_edges

    circuit_data["cleaned_graph"] = valid_graph
    circuit_data["plain_graph"] = graph

    # EMF positive is always in loop #1
    circuit_data["emf_pins"] = {"+": emf_pos_name, "-": emf_neg_name}

    # compute the independent loops on the graph
    circuit_data["loop_unique_edges"] = []
    if circuit_data["number_loops"] > 1:
        encountered_edges = set()
        for indx in range(circuit_data["number_loops"]):
            new_edges = []
            for edg in flow_edges[indx]:
                if edg not in encountered_edges:
                    new_edges.append(edg)
                    encountered_edges.add(edg)
            circuit_data["loop_unique_edges"].append(new_edges)
    else:
        circuit_data["loop_unique_edges"] = list(flow_edges)

    return circuit_data


def load_class(class_path: str) -> type:
    *module_path, class_name = class_path.split(".")
    module = import_module(".".join(module_path))
    return getattr(module, class_name)


def get_unique_connections(instances, feature):
    source_feat, target_feat = feature
    m0_connector_ends = [
        (tuple(source), tuple(target))
        for source, target in zip(instances[source_feat._id], instances[target_feat._id])
    ]

    m0_connector_ends = tuple(
        {(source, target) for source, target in m0_connector_ends if source[:-1] != target[:-1]}
    )

    unique_connections = {
        (source[-2:], target[-2:]): (source, target) for source, target in m0_connector_ends
    }
    return tuple((source, target) for source, target in unique_connections.values())
    # return tuple((source, target) for source, target in m0_connector_ends)


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


def make_circuit_interpretations(
    m1_model: pm.Model,
    required_interpretations: int = 1,
    max_tries: int = 20,
    print_exceptions: bool = True,
    on_attempt: ty.Callable = None,
) -> tuple:
    """Returns a tuple of data on valid randomly generated M0 interpretations."""
    num_tries = 0
    interpretations = []
    circuit_pkg = m1_model.ownedElement["Circuit Builder"]
    connector_feature = circuit_pkg.ownedElement["Circuit"].ownedMember["Part to Part"].endFeature
    while len(interpretations) < required_interpretations and num_tries < max_tries:
        num_tries += 1
        if on_attempt is not None:
            on_attempt()
        try:
            interpretation = random_generator_playbook(
                m1=m1_model,
                filtered_feat_packages=[circuit_pkg],
            )
            interpretation_data = get_circuit_data(interpretation, connector_feature)
            graph: nx.DiGraph = interpretation_data["cleaned_graph"]
            assert len(graph.nodes) > 0, "Graph has no nodes!"
            assert len(graph.edges) > 0, "Graph has no edges!"
            interpretation_data["problem"] = execute_interpretation(
                interpretation_data=interpretation_data,
                print_exceptions=print_exceptions,
            )
            interpretations += [interpretation_data]
        except:  # pylint: disable=bare-except  # noqa: E722
            if print_exceptions:
                print(
                    f">>> Failed to generate interpretation try {num_tries}!\n",
                    f"\n\tTRACEBACK:\n\t{traceback.format_exc()}",
                )
    return tuple(interpretations)


class CircuitComponent(om.Group):
    """An OpenMDAO Circuit for circuits of Diodes and Resistors."""

    # FIXME: this should be retrieved from the SysML model
    OM_COMPONENTS = {
        class_path.rsplit(".", 1)[-1][0]: load_class(class_path)
        for class_path in (
            "openmdao.test_suite.test_examples.test_circuit_analysis.Diode",
            "openmdao.test_suite.test_examples.test_circuit_analysis.Resistor",
            "openmdao.test_suite.test_examples.test_circuit_analysis.Node",
        )
    }

    def initialize(self):
        self.options.declare("interpretation_data", types=dict)
        self.options.declare("print_exceptions", False, types=bool)
        self.options.declare("run_baseline", False, types=bool)
        self.options.declare("electric_params", {}, types=dict)

    def setup(self):
        print_exceptions = self.options["print_exceptions"]

        if self.options["run_baseline"]:
            data = baseline_openmdao_example_circuit()
            electric_params = data["electric_params"]
        else:
            data = self.options["interpretation_data"]
            electric_params = self.options["electric_params"]

        digraph: nx.DiGraph = data["cleaned_graph"]

        # V, *_ = next(digraph.successors(data["emf_pins"]["+"]))
        grounded = [el for el, *_ in digraph.predecessors(data["emf_pins"]["-"])]

        elements = {
            element for element, polarity in digraph.nodes if not element.upper().startswith("EMF")
        }
        for element in elements:
            comp_cls = self.OM_COMPONENTS.get(element[0])
            if comp_cls is None:
                if print_exceptions:
                    print(f"{element} doesn't have class!")
                continue
            kwargs = {}
            params = electric_params.get(element, {})
            if params:
                if print_exceptions:
                    print(f"Setting params={params} for {element}")
            if element in grounded:
                kwargs["promotes_inputs"] = [("V_out", "Vg")]
            self.add_subsystem(f"{element}", comp_cls(**params), **kwargs)

        connectors = nx.DiGraph()
        connectors.add_edges_from([(src, tgt) for src, tgt in digraph.edges if src[0] != tgt[0]])
        self.node_names = node_names = []
        for node_id, comp in enumerate(nx.connected_components(connectors.to_undirected())):
            node_name = f"node_{node_id}"

            if data["emf_pins"]["-"] in comp:
                if print_exceptions:
                    print(
                        f"  > Not adding '{node_name}' because it is connected to"
                        f" the ground ({data['emf_pins']['-']})"
                    )
                continue
            node_names += [node_name]

            has_pos_emf = data["emf_pins"]["+"] in comp
            if has_pos_emf:
                self.source_node = node_name

            n_in = sum(1 for node in comp if node[1] == "Pos")
            n_out = sum(1 for node in comp if node[1] == "Neg")

            kwargs = dict(promotes_inputs=[("I_in:0", "I_in")]) if has_pos_emf else {}
            self.add_subsystem(
                node_name,
                self.OM_COMPONENTS["N"](n_in=n_in, n_out=n_out),
                **kwargs,
            )
            if print_exceptions:
                print(f"  > Adding '{node_name}' with {n_in} inputs and {n_out} outputs")
            indeces = {"in": 1 * has_pos_emf, "out": 0}
            elec_volt_pins = []
            for element, polarity in comp:
                if element.startswith("EMF"):
                    continue
                elem_dir = "out" if polarity == "Pos" else "in"
                node_dir = "out" if elem_dir == "in" else "in"
                elec_volt_pins += [f"{element}.V_{elem_dir}"]

                node_current = f"{node_name}.I_{node_dir}:{indeces[node_dir]}"
                self.connect(f"{element}.I", node_current)
                if print_exceptions:
                    print(f"  > Connecting currents: {element}.I --> {node_current}")
                indeces[node_dir] += 1
            if elec_volt_pins:
                try:
                    self.connect(f"{node_name}.V", elec_volt_pins)
                    if print_exceptions:
                        print(
                            f" >  Connecting voltages for node {node_name}.V --> {elec_volt_pins}"
                        )
                except:  # pylint: disable=bare-except  # noqa: E722
                    if print_exceptions:
                        print(f"  ! Could not connect: {node_name}.V --> {elec_volt_pins}")

        self.nonlinear_solver = om.NewtonSolver()
        self.linear_solver = om.DirectSolver()

        self.nonlinear_solver.options["iprint"] = 2 if print_exceptions else 0
        self.nonlinear_solver.options["maxiter"] = 10
        self.nonlinear_solver.options["solve_subsystems"] = True
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.linesearch.options["maxiter"] = 10
        self.nonlinear_solver.linesearch.options["iprint"] = 2

        return True


def execute_interpretation(interpretation_data: dict, print_exceptions=False):
    try:
        problem = interpretation_data.get("problem")
        if problem is None:
            circuit_name = "circuit"
            problem = om.Problem()
            problem.model.add_subsystem(
                circuit_name,
                CircuitComponent(
                    interpretation_data=interpretation_data, print_exceptions=print_exceptions
                ),
            )
            is_valid = problem.setup()
            if not is_valid:
                raise ValueError("  ! Interpretation does not produce a valid circuit!")
            if print_exceptions:
                print(" >> Successfully set up interpretation")
        try:
            circuit = getattr(problem.model, circuit_name)
            problem.set_val(f"{circuit_name}.I_in", 0.1)
            problem.set_val(f"{circuit_name}.Vg", 0)
            for node_name in circuit.node_names:
                problem.set_val(
                    f"{circuit_name}.{node_name}.V",
                    (10.0 if node_name == circuit.source_node else 0.1),
                )
            problem.run_model()
            if print_exceptions:
                print("  + Successfully ran interpretation!\n")
        except Exception:  # pylint: disable=broad-except
            if print_exceptions:
                print(f"  ! Failed to run interpretation!\n\n{traceback.format_exc()}\n")
    except Exception:  # pylint: disable=broad-except
        if print_exceptions:
            print(f"  ! Failed to setup interpretation!\n\n{traceback.format_exc()}\n")
    return problem


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


class CircuitUI(DockPop):
    """A user interface for interacting with the Circuit Builder example."""

    DEFAULT_LAYOUT = {
        "type": "split-area",
        "orientation": "horizontal",
        "children": [
            {"type": "tab-area", "widgets": [0], "currentIndex": 0},
            {"type": "tab-area", "widgets": [1, 3, 2], "currentIndex": 0},
        ],
        "sizes": [0.20, 0.80],
    }

    panels: DockBox = trt.Instance(
        DockBox, kw=dict(description="Circuit Generator", dock_layout=DEFAULT_LAYOUT)
    )

    circuit_model: pm.Model = trt.Instance(pm.Model, allow_none=True)
    interpretation: dict = trt.Dict()

    instance_generator: CircuitGenerator = trt.Instance(CircuitGenerator, args=())
    graph_plotter: CircuitPlotter = trt.Instance(CircuitPlotter, args=())
    connections: ipyw.Output = trt.Instance(ipyw.Output, kw=dict(layout=dict(width="100%")))
    n2: ipyw.Output = trt.Instance(ipyw.Output, kw=dict(layout=dict(width="100%")))

    @trt.default("circuit_model")
    def _get_circuit_model(self) -> pm.Model:
        return self.instance_generator.circuit_model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        generator = self.instance_generator

        if self.circuit_model is None:
            self.circuit_model = self.instance_generator.circuit_model

        panels = {
            "Generator": generator,
            "Graph": self.graph_plotter,
            "Results": self.connections,
            "N2": self.n2,
        }
        self.panels.children = tuple(panels.values())
        self.children = [self.panels]

        for name, widget in panels.items():
            widget.add_traits(description=trt.Unicode(default_value=name).tag(sync=True))

        trt.link((self, "circuit_model"), (generator, "circuit_model"))
        trt.link((self, "interpretation"), (generator, "selected_interpretation"))

    @trt.observe("interpretation")
    def _update_graph(self, *_):
        graph = self.interpretation.get("cleaned_graph")
        if isinstance(graph, nx.DiGraph):
            self.graph_plotter.graph = graph

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
