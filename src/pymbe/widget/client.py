from collections import Counter

import ipywidgets as ipyw
import traitlets as trt

from ..client import SysML2Client

__all__ = ("SysML2ClientWidget",)


@ipyw.register
class SysML2ClientWidget(SysML2Client, ipyw.GridspecLayout):
    """An ipywidget to interact with a SysML v2 API."""

    description = trt.Unicode("API Client").tag(sync=True)

    # file_uploader: ipyw.FileUpload = trt.Instance(ipyw.FileUpload)
    host_url_input: ipyw.Text = trt.Instance(ipyw.Text)
    host_port_input: ipyw.IntText = trt.Instance(ipyw.IntText)
    project_selector: ipyw.Dropdown = trt.Instance(ipyw.Dropdown)
    commit_selector: ipyw.Dropdown = trt.Instance(ipyw.Dropdown)
    download_elements: ipyw.Button = trt.Instance(ipyw.Button)
    progress_bar: ipyw.IntProgress = trt.Instance(ipyw.IntProgress)

    def __init__(self, n_rows=4, n_columns=12, **kwargs):
        super().__init__(n_rows=n_rows, n_columns=n_columns, **kwargs)

    def _ipython_display_(self, **kwargs):
        super()._ipython_display_(**kwargs)
        self._set_layout()

    @trt.validate("children")
    def _set_children(self, proposal):
        children = proposal.value
        if children:
            return children
        return [
            self.host_url_input,
            self.host_port_input,
            self.project_selector,
            self.commit_selector,
            self.download_elements,
            self.progress_bar,
        ]

    def _set_layout(self):
        layout = self.layout
        layout.overflow_y = "hidden"
        layout.width, layout.height = ["auto"] * 2

        idx = self.n_columns - 1
        self[0, :idx] = self.host_url_input
        self[0, idx:] = self.host_port_input
        self[1, :idx] = self.project_selector
        self[2, :idx] = self.commit_selector
        self[1:3, idx:] = self.download_elements
        self[3, :] = self.progress_bar

        for widget in (
            self.host_url_input,
            self.host_port_input,
            self.project_selector,
            self.commit_selector,
            self.download_elements,
            self.progress_bar,
        ):
            widget.layout.height = "95%"
            widget.layout.width = "auto"
            widget.layout.overflow_y = "hidden"
            widget.layout.overflow_x = "hidden"
            widget.layout.min_height = None
            widget.layout.max_height = None
            widget.layout.max_width = None
            widget.layout.min_width = None

        self.layout = layout

    @trt.default("host_url_input")
    def _make_host_url_input(self):
        input_box = ipyw.Text(
            default_value=self.host_url,
            description="Server:",
            description_tooltip="The URL and port of the SysML v2 Server",
        )
        trt.link(
            (self, "host_url"),
            (input_box, "value"),
        )
        return input_box

    @trt.default("host_port_input")
    def _make_host_port_input(self):
        input_box = ipyw.IntText(
            default_value=self.host_port,
            min=1,
            max=65535,
            layout=dict(max_width="6rem"),
        )
        trt.link(
            (self, "host_port"),
            (input_box, "value"),
        )
        return input_box

    @trt.default("project_selector")
    def _make_project_selector(self):
        selector = ipyw.Dropdown(
            description="Project:",
            options=self._get_project_options(),
        )
        trt.link((selector, "value"), (self, "selected_project"))
        return selector

    @trt.default("commit_selector")
    def _make_commit_selector(self):
        selector = ipyw.Dropdown(
            description="Commit:",
            options=self._get_commit_selector_options() if self.project_selector.options else {},
        )
        trt.link((selector, "value"), (self, "selected_commit"))
        return selector

    @trt.default("download_elements")
    def _make_download_elements_button(self):
        button = ipyw.Button(
            icon="cloud-download",
            tooltip="Fetch elements from remote host.",
            layout=dict(max_width="6rem"),
        )
        button.on_click(self._download_elements)
        return button

    @trt.default("progress_bar")
    def _make_progress_bar(self):
        progress_bar = ipyw.IntProgress(
            description="Loading:",
            min=0,
            max=4,
            step=1,
            value=0,
        )
        progress_bar.style.bar_color = "gray"
        progress_bar.layout.visibility = "hidden"
        return progress_bar

    @trt.observe("projects")
    def _update_apis(self, *_):
        self.project_selector.options = self._get_project_options()

    @trt.observe("selected_project")
    def _update_commit_options(self, *_):
        if self.project_selector.options:
            self.commit_selector.options = self._get_commit_selector_options()

    def _get_commit_selector_options(self):
        return {
            f"""{data["timestamp"].strftime("%a %b %d %H:%M:%S %Y")} <{id_}>""": id_
            for id_, data in self._get_project_commits().items()
        }

    def _download_elements(self, *_):
        progress = self.progress_bar
        progress.value = 0
        progress.layout.visibility = "visible"

        progress.value += 1

        super()._download_elements()
        progress.value += 1

        progress.value = progress.max
        progress.layout.visibility = "hidden"

    def _get_project_options(self):
        project_name_instances = Counter(project["name"] for project in self.projects.values())

        return {
            data["name"]
            + (
                f""" ({data["created"].strftime("%Y-%m-%d %H:%M:%S")})"""
                if project_name_instances[data["name"]] > 1
                else ""
            ): id_
            for id_, data in sorted(
                self.projects.items(),
                key=lambda x: x[1]["name"],
            )
        }
