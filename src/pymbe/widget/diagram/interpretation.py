import ipywidgets as ipyw
import traitlets as trt
from ipyelk import Diagram, ElementLoader
from ipyelk.elements import Label, Node, Port
from ipyelk.elements import layout_options as opt

from ...graph import SysML2LabeledPropertyGraph
from ...interpretation.interp_playbooks import random_generator_playbook
from ...interpretation.interpretation import repack_instance_dictionaries
from ...model import InstanceDictType
from ..core import BaseWidget
from .part_diagram import PartDiagram
from .tools import BUTTON_ICONS

NODE_LAYOUT_OPTIONS = {
    "org.eclipse.elk.portLabels.placement": "INSIDE",
    "org.eclipse.elk.nodeSize.constraints": "NODE_LABELS PORTS PORT_LABELS MINIMUM_SIZE",
    "org.eclipse.elk.nodeLabels.placement": "H_CENTER V_CENTER",
}

__all__ = ("M0Viewer",)


class M0Viewer(ipyw.Box, BaseWidget):
    """A viewer for M0 interpretations"""

    description: str = trt.Unicode("M0 Diagram").tag(sync=True)
    icon_class: str = trt.Unicode("jp-ExtensionIcon").tag(sync=True)

    diagram: Diagram = trt.Instance(Diagram)
    loader: ElementLoader = trt.Instance(
        ElementLoader,
        kw=dict(
            default_node_opts=opt.OptionsWidget(
                options=[
                    opt.Direction(value="RIGHT"),
                    opt.HierarchyHandling(),
                ]
            ).value,
        ),
    )
    lpg: SysML2LabeledPropertyGraph = trt.Instance(
        SysML2LabeledPropertyGraph,
        args=(),
        help="The LPG of the project currently loaded.",
    )
    interpretation: InstanceDictType = trt.Dict()

    package_selector: ipyw.SelectMultiple = trt.Instance(ipyw.SelectMultiple, args=())

    port_size: int = trt.Integer(15)

    @trt.validate("children")
    def _validate_children(self, proposal: trt.Bunch):
        children = proposal.value
        if children:
            return children
        return [self.diagram]

    @trt.default("diagram")
    def _make_diagram(self):
        diagram = Diagram(layout=dict(height="100%"))
        toolbar = diagram.toolbar
        style, *buttons, progress_bar, close_btn = toolbar.children
        for button in buttons:
            description = button.description
            icon = BUTTON_ICONS.get(description)
            if icon:
                button.description = ""
                button.tooltip = description
                button.icon = icon
                button.layout.width = "40px"

        btn_kwargs = dict(layout=dict(width="40px"))

        regen = ipyw.Button(
            icon="redo",
            tooltip="Generate new random M0 interpretation",
            **btn_kwargs,
        )
        regen.on_click(self._generate_random_interpretation)

        refresh = ipyw.Button(
            icon="retweet",
            tooltip="Refresh Diagram",
            **btn_kwargs,
        )
        refresh.on_click(self._update_for_new_interpretation)

        buttons += [regen, refresh]

        toolbar.children = [style, *buttons, self.package_selector, progress_bar, close_btn]

        return diagram

    @trt.default("layout")
    def _make_layout(self):
        return dict(height="100%")

    def _get_model_packages(self):
        if not self.model:
            return {}
        return dict(
            sorted(
                (package.name, package) for package in self.model.ownedMetatype.get("Package", [])
            )
        )

    def update(self, *_):
        self.package_selector.options = self._get_model_packages()

    def _generate_random_interpretation(self, *_):
        with self.log_out:
            self.interpretation = random_generator_playbook(
                m1=self.lpg,
                filtered_feat_packages=self.package_selector.value,
            )

    @trt.observe("interpretation")
    def _update_for_new_interpretation(self, *_):
        part_diagram = PartDiagram()
        repacked = repack_instance_dictionaries(self.interpretation, self.model)

        def is_draw_kind(entry, kind):
            if not entry.value:
                return False
            draw_kind = next(iter(entry.value)).owning_entry.draw_kind
            return draw_kind and kind in draw_kind

        elk_nodes = {}

        # part_definitions = [
        #     entry for entry in repacked if entry.base._metatype == "PartDefinition"
        # ]
        # TODO: Figure out if we need to do the part definitions
        # parts = part_diagram.add_child(Node(
        #     labels=[
        #         Label(text="Parts"),
        #     ],
        #     layoutOptions=NODE_LAYOUT_OPTIONS,
        # ))
        # for entry in part_definitions:
        #     for sequence in entry.value:
        #         parent_node = parts
        #         for instance in sequence.instances:
        #             elk_node = elk_nodes.get(instance)
        #             if elk_node is None:
        #                 elk_nodes[instance] = elk_node = parent_node.add_child(
        #                     Node(
        #                         labels=[
        #                             # TODO: Find out how to represent type
        #                             Label(text=f"`{entry.base.label}`"),
        #                             Label(text=instance.name),
        #                         ],
        #                         layoutOptions=NODE_LAYOUT_OPTIONS,
        #                     ),
        #                 )
        #             parent_node = elk_node

        nodes = [entry for entry in repacked if is_draw_kind(entry, "Rectangle")]
        for interpreted_node in nodes:
            for sequence in interpreted_node.value:
                # parent_instance, *remaining_instances = sequence.instances
                # parent_node = elk_nodes[parent_instance]
                # for instance in remaining_instances:
                parent_node = part_diagram
                for instance in sequence.instances:
                    elk_node = elk_nodes.get(instance)
                    if elk_node is None:
                        elk_nodes[instance] = elk_node = parent_node.add_child(
                            Node(
                                labels=[
                                    # TODO: Find out how to represent type
                                    # Label(text=f"`{interpreted_node.base.label}`"),
                                    Label(text=instance.name),
                                ],
                                layoutOptions=NODE_LAYOUT_OPTIONS,
                            ),
                        )
                    parent_node = elk_node

        ports = [entry.value for entry in repacked if is_draw_kind(entry, "Port")]
        elk_ports = {}
        for port_set in ports:
            for sequence in port_set:
                *_, node, port = sequence.instances
                elk_ports[port] = elk_nodes[node].add_port(
                    port=Port(
                        layoutOptions={
                            "org.eclipse.elk.port.borderOffset": f"-{self.port_size}",
                        },
                        labels=[Label(text=port.name.split("#")[-1])],
                        width=self.port_size,
                        height=self.port_size,
                    ),
                )

        lines = [entry.value for entry in repacked if is_draw_kind(entry, "Line")]
        for line in lines:
            for connection in line:
                src, tgt = connection.get_line_ends()
                part_diagram.add_edge(
                    source=elk_ports[src.instances[-1]],
                    target=elk_ports[tgt.instances[-1]],
                )

        diagram = self.diagram
        diagram.source = self.loader.load(part_diagram)
        diagram.style = part_diagram.style
        diagram.view.symbols = part_diagram.symbols
