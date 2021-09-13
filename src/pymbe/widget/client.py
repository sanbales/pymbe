import json
from collections import Counter

import ipywidgets as ipyw
import traitlets as trt
from wxyz.html import File, FileBox

from ..client import APIClient
from ..model import Model
from .core import BaseWidget

__all__ = ("APIClientWidget", "FileLoader")


@ipyw.register
class FileLoader(FileBox, BaseWidget):
    """A simple UI for loading SysML models from disk."""

    closable: bool = trt.Bool(True).tag(sync=True)
    description: str = trt.Unicode("File Loader").tag(sync=True)
    icon_class: str = trt.Unicode("jp-JsonIcon").tag(sync=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accept = ["json"]
        self.multiple = False

    def update(self, *_):
        self.children = []

    def _load_model(self, change: trt.Bunch):
        with self.log_out:
            model_file, *other = self.children
            assert not other, f"Should only have one file, but also found {other}"
            self.model = Model.load(
                elements=json.loads(change.new),
                name=".".join(model_file.name.split(".")[:-1]),
            )

    @trt.observe("children")
    def _update_model(self, change: trt.Bunch):
        with self.log_out:
            if isinstance(change.old, (list, tuple)) and change.old:
                old, *_ = change.old
                if isinstance(old, File):
                    old.unobserve(self._load_model)
            if isinstance(change.new, (list, tuple)) and change.new:
                new, *_ = change.new
                new.observe(self._load_model, "value")


@ipyw.register
class APIClientWidget(APIClient, ipyw.GridspecLayout):
    """An ipywidget to interact with a SysML v2 API."""

    closable: bool = trt.Bool(True).tag(sync=True)
    description: str = trt.Unicode("API Client").tag(sync=True)
    icon_class: str = trt.Unicode("jp-DownloadIcon").tag(sync=True)

    model: Model = trt.Instance(Model, allow_none=True)

    # file_uploader: ipyw.FileUpload = trt.Instance(ipyw.FileUpload)
    host_url_input: ipyw.Text = trt.Instance(ipyw.Text)
    host_port_input: ipyw.IntText = trt.Instance(ipyw.IntText)
    project_selector: ipyw.Dropdown = trt.Instance(ipyw.Dropdown)
    commit_selector: ipyw.Dropdown = trt.Instance(ipyw.Dropdown)
    download_model: ipyw.Button = trt.Instance(ipyw.Button)
    progress_bar: ipyw.FloatProgress = trt.Instance(ipyw.FloatProgress)

    log_out: ipyw.Output = trt.Instance(ipyw.Output, args=())

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
            self.download_model,
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
        self[1:3, idx:] = self.download_model
        self[3, :] = self.progress_bar

        for widget in (
            self.host_url_input,
            self.host_port_input,
            self.project_selector,
            self.commit_selector,
            self.download_model,
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
    def _make_host_url_input(self) -> ipyw.Text:
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
    def _make_host_port_input(self) -> ipyw.IntText:
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
    def _make_project_selector(self) -> ipyw.Dropdown:
        selector = ipyw.Dropdown(
            description="Project:",
            options=self._get_project_options(),
        )
        trt.link((selector, "value"), (self, "selected_project"))
        return selector

    @trt.default("commit_selector")
    def _make_commit_selector(self) -> ipyw.Dropdown:
        selector = ipyw.Dropdown(
            description="Commit:",
            options=self._get_commit_selector_options() if self.project_selector.options else {},
        )
        trt.link((selector, "value"), (self, "selected_commit"))
        return selector

    @trt.default("download_model")
    def _make_download_model_button(self) -> ipyw.Button:
        button = ipyw.Button(
            icon="cloud-download",
            tooltip="Fetch elements from remote host.",
            layout=dict(max_width="6rem"),
        )
        button.on_click(self._download_model)
        return button

    @trt.default("progress_bar")
    def _make_progress_bar(self) -> ipyw.FloatProgress:
        progress_bar = ipyw.FloatProgress(
            description="Loading:",
            min=0,
            max=1,
            value=0,
        )
        progress_bar.style.bar_color = "gray"
        progress_bar.layout.visibility = "hidden"
        progress_bar.description_tooltip = "Loading model..."
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

    def _download_model(self, *_):
        with self.log_out:
            progress = self.progress_bar

            progress.style.bar_color = "gray"
            progress.value = 0
            progress.layout.visibility = "visible"

            progress.value += 0.1

            step_per_element = 0.001
            data = dict(page=0)

            def on_page():
                data["page"] += 1
                page = data["page"]
                elements = page * self.page_size
                if (elements * step_per_element) > (0.8 * progress.max):
                    progress.max *= 1.5
                progress.description_tooltip = f"Retrieved page {page} ({elements} model elements)"
                progress.value += self.page_size * step_per_element

            progress.description_tooltip = "Downloading model..."
            model = self.get_model(on_page=on_page)

            if not model or not model.elements:
                progress.description_tooltip = "Failed to download the model..."
                progress.style = "danger"
                raise ValueError(f"Could not download model from: {self.elements_url}")

            progress.description_tooltip = "Finished downloading the model..."
            self.model = model

            progress.value = progress.max
            progress.layout.visibility = "hidden"

    def _get_project_options(self):
        project_name_instances = Counter(project["name"] for project in self.projects.values())

        return {
            data["name"]
            + (
                f""" ({data["created"].strftime("%Y-%m-%d %H:%M:%S")} UTC)"""
                if project_name_instances[data["name"]] > 1
                else ""
            ): id_
            for id_, data in sorted(
                self.projects.items(),
                key=lambda x: (x[1]["name"], x[1]["created"]),
            )
        }
