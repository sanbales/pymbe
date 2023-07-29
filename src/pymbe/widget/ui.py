import ipywidgets as ipyw
import traitlets as trt
from ipylab import JupyterFrontEnd

from .containment import ContainmentTree
from .diagram import M1Viewer
from .inspector import ElementInspector


class UI(JupyterFrontEnd):
    """A JupyterLab user interface for the integrated widget"""

    # widgets
    tree: ContainmentTree = trt.Instance(ContainmentTree, args=())
    inspector: ElementInspector = trt.Instance(ElementInspector, args=())
    m1_viewer: M1Viewer = trt.Instance(M1Viewer, args=())

    # links
    log_out_links: list = trt.List()
    lpg_links: list = trt.List()
    model_links: list = trt.List()
    selector_links: list = trt.List()

    # config parameters
    diagram_height: int = trt.Int(default_value=65, min=25, max=100)

    log_out: ipyw.Output = trt.Instance(ipyw.Output, args=())

    @trt.default("tree")
    def _make_tree(self) -> ContainmentTree:
        return ContainmentTree(app=self)

    def __init__(self, host_url, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.description = "SysML Model"

        all_widgets = self.tree, self.inspector, self.m1_viewer

        self.log_out_links = [
            trt.link((self, "log_out"), (widget, "log_out")) for widget in all_widgets
        ]

        first, *other_widgets = all_widgets
        self.model_links = [
            trt.link(
                (first, "model"),
                (widget, "model"),
            )
            for widget in other_widgets
        ]
        self.selector_links = [
            trt.link(
                (first, "selected"),
                (widget, "selected"),
            )
            for widget in other_widgets
        ]

        self.add_panels()
        # TODO: find a way to avoid doing these three lines
        self._update_diagram_height()

    @trt.observe("diagram_height")
    def _update_diagram_height(self, *_):
        self.m1_viewer.layout.height = f"{self.diagram_height}vh"

    def add_panels(self):
        self.shell.add(self.tree, "left")
        self.shell.add(self.inspector, "main", {"mode": "split-right"})
        self.shell.add(self.m1_viewer, "main", {"mode": "split-right"})
