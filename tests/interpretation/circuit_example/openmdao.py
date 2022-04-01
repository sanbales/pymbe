import traceback
from importlib import import_module

import networkx as nx

import openmdao.api as om

__all__ = (
    "CircuitComponent",
    "execute_interpretation",
)


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


def load_class(class_path: str) -> type:
    *module_path, class_name = class_path.split(".")
    module = import_module(".".join(module_path))
    return getattr(module, class_name)


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

        self._add_nodes(digraph=digraph, emf_pins=data["emf_pins"])

        self.nonlinear_solver = om.NewtonSolver()
        self.linear_solver = om.DirectSolver()

        self.nonlinear_solver.options["iprint"] = 2 if print_exceptions else 0
        self.nonlinear_solver.options["maxiter"] = 10
        self.nonlinear_solver.options["solve_subsystems"] = True
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.linesearch.options["maxiter"] = 10
        self.nonlinear_solver.linesearch.options["iprint"] = 2

        return True

    def _add_nodes(self, digraph: nx.DiGraph, emf_pins: dict, print_exceptions: bool = False):
        connectors = nx.DiGraph()
        connectors.add_edges_from([(src, tgt) for src, tgt in digraph.edges if src[0] != tgt[0]])
        self.node_names = node_names = []
        for node_id, comp in enumerate(nx.connected_components(connectors.to_undirected())):
            node_name = f"node_{node_id}"

            if emf_pins["-"] in comp:
                if print_exceptions:
                    print(
                        f"  > Not adding '{node_name}' because it is connected to"
                        f" the ground ({emf_pins['-']})"
                    )
                continue
            node_names += [node_name]

            has_pos_emf = emf_pins["+"] in comp
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
