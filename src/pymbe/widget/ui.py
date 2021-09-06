import ipywidgets as ipyw
import traitlets as trt
from wxyz.lab import DockBox, DockPop

from .containment import ContainmentTree
from .diagram import M0Viewer, M1Viewer
from .inspector import ElementInspector


@ipyw.register
class UI(DockBox):
    """A user interface for the integrated widget"""

    # widgets
    tree: ContainmentTree = trt.Instance(ContainmentTree, args=())
    inspector: ElementInspector = trt.Instance(ElementInspector, args=())
    m0_viewer: M0Viewer = trt.Instance(M0Viewer, args=())
    m1_viewer: M1Viewer = trt.Instance(M1Viewer, args=())

    # links
    lpg_links: list = trt.List()
    model_links: list = trt.List()
    selector_links: list = trt.List()

    # config parameters
    diagram_height: int = trt.Int(default_value=65, min=25, max=100)

    log_out: ipyw.Output = trt.Instance(ipyw.Output, args=())

    def __init__(self, host_url, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.description = "SysML Model"

        self.tree.client.host_url = host_url

        self.children = [
            self.tree,
            self.inspector,
            self.m1_viewer,
            self.m0_viewer,
        ]

        self.dock_layout = dict(
            type="split-area",
            orientation="horizontal",
            children=[
                dict(type="tab-area", widgets=[0], currentIndex=0),
                dict(
                    type="tab-area",
                    widgets=[1, 2, 3],
                    currentIndex=0,
                ),
            ],
            sizes=[0.22, 0.78],
        )

        # TODO: find a way to avoid doing these three lines
        self._update_diagram_height()

        all_widgets = self.inspector, self.m1_viewer, self.m0_viewer

        for widget in all_widgets:
            trt.link((self, "log_out"), (widget, "log_out"))

        self.model_links = [
            trt.link(
                (self.tree, "model"),
                (widget, "model"),
            )
            for widget in all_widgets
        ]
        self.selector_links = [
            trt.link(
                (self.tree, "selected"),
                (widget, "selected"),
            )
            for widget in all_widgets
        ]
        self.lpg_links = [
            trt.link(
                (self.m1_viewer, "lpg"),
                (self.m0_viewer, "lpg"),
            )
        ]

    @trt.observe("diagram_height")
    def _update_diagram_height(self, *_):
        self.m0_viewer.layout.height = f"{self.diagram_height}vh"
        self.m1_viewer.layout.height = f"{self.diagram_height}vh"

    @classmethod
    def new(cls, host_url: str) -> DockPop:
        sysml_ui = cls(host_url=host_url)
        dock_pop = DockPop([sysml_ui], description="SysML Model")

        def _add(widget: ipyw.DOMWidget, mode="split-right"):
            """Add the example preview in a new JupyterLab DockPanel tab"""
            dock_pop.mode = mode
            dock_pop.children += (widget,)

        for child in sysml_ui.children:
            if "add_widget" in child.trait_names():
                child.add_widget = _add

        return dock_pop
