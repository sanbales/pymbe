import ipywidgets as ipyw
import traitlets as trt
import typing as ty

import ipyelk as elk


class CenterButton(elk.tools.ToolButton):

    def handler(self, *_):
        diagram = self.app.diagram
        diagram.center(retain_zoom=True)


class FitButton(elk.tools.ToolButton):

    def handler(self, *_):
        diagram = self.app.diagram
        diagram.fit(padding=50)


class Toolbar(elk.toolbar.Toolbar):

    elk_app: elk.Elk = trt.Instance(elk.Elk)

    fit: FitButton = trt.Instance(FitButton)
    center: CenterButton = trt.Instance(CenterButton)

    toolbar_buttons: list = trt.List(trait=trt.Instance(ipyw.Button))
    toolbar_accordion: ty.Dict[str, ipyw.Widget] = trt.Dict(
        key_trait=trt.Unicode(),
        value_trait=trt.Instance(ipyw.Widget),
    )

    @trt.default("center")
    def _make_center_button(self) -> CenterButton:
        return CenterButton(
            app=self.elk_app,
            description="",
            icon="compress",
            layout=dict(height="40px", width="40px"),
            tooltip="Center Diagram",
        )

    @trt.default("fit")
    def _make_fit_button(self) -> FitButton:
        return FitButton(
            app=self.elk_app,
            description="",
            icon="expand-arrows-alt",
            layout=dict(height="40px", width="40px"),
            tooltip="Fit Diagram",
        )

    @trt.default("filter_to_path")
    def _make_filter_to_path_button(self) -> ipyw.Button:
        button = ipyw.Button(
            description="",
            icon="project-diagram",  # share-alt
            tooltip="Filter To Path",
            layout=dict(height="40px", width="40px")
        )
        button.on_click(self._update_diagram_graph)
        return button

    @trt.default("filter_by_dist")
    def _make_filter_by_dist_button(self) -> ipyw.Button:
        button = ipyw.Button(
            description="",
            icon="sitemap",  # hubspot
            tooltip="Filter by Distance",
            layout=dict(height="40px", width="40px")
        )
        button.on_click(self._update_diagram_graph)
        return button

    @trt.default("update_diagram")
    def _make_update_diagram_button(self) -> ipyw.Button:
        button = ipyw.Button(
            description="",
            icon="retweet",
            tooltip="Update diagram",
            layout=dict(height="40px", width="40px")
        )
        button.on_click(self._update_diagram_graph)
        return button

    @trt.default("toolbar_buttons")
    def _make_toolbar_buttons(self):
        return [self.fit, self.center]

    @trt.default("toolbar_accordion")
    def _make_toolbar_accordion(self):
        return {
            "Layout": self.elk_layout,
        }

    @trt.observe("toolbar_buttons", "toolbar_accordion")
    def _update_toolbar(self, *_):
        self.layout.width = "auto"
        self.commands = [self._make_command_palette()]

    def _make_command_palette(self) -> ipyw.VBox:
        titles, widgets = zip(*self.toolbar_accordion.items())
        titles = {
            idx: title
            for idx, title in enumerate(titles)
        }
        return ipyw.VBox(
            [
                ipyw.HBox(self.toolbar_buttons),
                ipyw.Accordion(
                    _titles=titles,
                    children=widgets,
                    selected_index=None,
                ),
            ],
        )

    def _update_diagram_toolbar(self):
        # Append elements to the elk_app toolbar
        diagram = self.diagram
        accordion = {**diagram.toolbar_accordion}

        sub_accordion = ipyw.Accordion(
            _titles={0: "Node Types", 1: "Edge Types"},
            children=[
                self.node_type_selector,
                self.edge_type_selector,
            ],
            selected_index=None,
        )
        accordion.update({
            # TODO: enable this after the functionality is complete
            # "Reverse Edges": self.edge_type_reverser,
            "Filter": ipyw.VBox([
                self.path_directionality,
                ipyw.Label("Shortest Path:"),
                ipyw.HBox([self.filter_to_path, ]),
                ipyw.Label("Distance:"),
                ipyw.HBox([self.filter_by_dist, self.max_distance]),
                sub_accordion,
            ]),
        })

        buttons = [*diagram.toolbar_buttons]
        buttons += [self.update_diagram]

        with diagram.hold_trait_notifications():
            diagram.toolbar_accordion = accordion
            diagram.toolbar_buttons = buttons
