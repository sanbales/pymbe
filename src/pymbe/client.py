import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union
from warnings import warn

import ipywidgets as ipyw
import requests
import traitlets as trt
from dateutil import parser
from ipywidgets.widgets.trait_types import TypedTuple

from .label import get_label
from .model import Model

TIMEZONES = {
    "CEST": "UTC+2",
    "CET": "UTC+1",
    "EDT": "UTC-4",
    "EST": "UTC-5",
    "CDT": "UTC-5",
    "CST": "UTC-6",
    "MDT": "UTC-6",
    "MST": "UTC-7",
    "PDT": "UTC-7",
    "PST": "UTC-8",
    "AKDT": "UTC-8",
    "AKST": "UTC-9",
    "HDT": "UTC-9",
    "HST": "UTC-10",
    "AoE": "UTC-12",
    "SST": "UTC-11",
    "AST": "UTC-4",
    "ChST": "UTC+10",
    "WAKT": "UTC+12",
}


class SysML2Client(trt.HasTraits):
    """
        A traitleted SysML v2 API Client.

    ..todo:
        - Add ability to use element download pagination.

    """

    model: Model = trt.Instance(Model, allow_none=True)

    host_url = trt.Unicode(
        default_value="http://localhost",
    )

    host_port = trt.Integer(
        default_value=9000,
        min=1,
        max=65535,
    )

    page_size = trt.Integer(
        default_value=5000,
        min=1,
    )

    paginate = trt.Bool(default_value=True)

    folder_path: Path = trt.Instance(Path, allow_none=True)
    json_files: Tuple[Path] = TypedTuple(trt.Instance(Path))
    json_file: Path = trt.Instance(Path, allow_none=True)

    selected_project: str = trt.Unicode(allow_none=True)
    selected_commit: str = trt.Unicode(allow_none=True)

    projects = trt.Dict()

    name_hints = trt.Dict()

    log_out: ipyw.Output = trt.Instance(ipyw.Output, args=())

    _next_url_regex = re.compile(r'<(http://.*)>; rel="next"')

    @trt.default("projects")
    def _make_projects(self):
        def process_project_safely(project) -> dict:
            # protect against projects that can't be parsed
            try:
                name_with_date = project["name"]
                name = " ".join(name_with_date.split()[:-6])
                timestamp = " ".join(name_with_date.split()[-6:])
                created = self._parse_timestamp(timestamp)
            except ValueError:
                # TODO: revise this when the API server changes the project name
                return dict()
            return dict(
                created=created,
                full_name=name_with_date,
                name=name,
            )

        try:
            projects = self._retrieve_data(self.projects_url)
        except Exception as exc:  # pylint: disable=broad-except
            warn(f"Could not retrieve projects from {self.projects_url}.\n{exc}")
            return {}

        results = {project["@id"]: process_project_safely(project) for project in projects}

        return {
            project_id: project_data
            for project_id, project_data in results.items()
            if project_data
        }

    @trt.observe("host_url", "host_port")
    def _update_api_configuration(self, *_):
        self.projects = self._make_projects()

    @trt.observe("selected_commit")
    def _update_elements(self, *_, elements=None):
        if not (self.selected_commit or elements):
            return
        elements = elements or []
        self.model = Model.load(
            elements=elements,
            name=f"""{
                self.projects[self.selected_project]["name"]
            } ({self.host})""",
            source=self.elements_url,
        )
        for element in self.model.elements.values():
            if "label" not in element._derived:
                element._derived["label"] = get_label(element)

    @trt.observe("folder_path")
    def _update_json_files(self, *_):
        if self.folder_path.exists():
            self.json_files = tuple(self.folder_path.glob("*.json"))

    @trt.observe("json_file")
    def _update_elements_from_file(self, change: trt.Bunch = None):
        if change is None:
            return
        if change.new != change.old and change.new.exists():
            self.model = Model.load_from_file(self.json_file)

    @property
    def host(self):
        return f"{self.host_url}:{self.host_port}"

    @property
    def projects_url(self):
        return f"{self.host}/projects"

    @property
    def commits_url(self):
        return f"{self.projects_url}/{self.selected_project}/commits"

    @property
    def elements_url(self):
        if not self.paginate:
            warn(
                "By default, disabling pagination still retrieves 100 "
                "records at a time!  True pagination is not supported yet."
            )
        if not self.selected_project:
            raise SystemError("No selected project!")
        if not self.selected_commit:
            raise SystemError("No selected commit!")
        arguments = f"?page[size]={self.page_size}" if self.page_size else ""
        return f"{self.commits_url}/{self.selected_commit}/elements{arguments}"

    @lru_cache
    def _retrieve_data(self, url: str) -> List[Dict]:
        """Retrieve model data from a URL using pagination"""
        result = []
        while url:
            response = requests.get(url)

            if not response.ok:
                raise requests.HTTPError(
                    f"Failed to retrieve elements from '{url}', reason: {response.reason}"
                )

            result += response.json()

            link = response.headers.get("Link")
            if not link:
                break

            urls = self._next_url_regex.findall(link)
            url = None
            if len(urls) == 1:
                url = urls[0]
            elif len(urls) > 1:
                raise SystemError(f"Found multiple 'next' pagination urls: {urls}")
        return result

    @staticmethod
    def _parse_timestamp(timestamp: str) -> datetime:
        if isinstance(timestamp, datetime):
            return timestamp
        return parser.parse(timestamp, tzinfos=TIMEZONES).astimezone(timezone.utc)

    def _get_project_commits(self):
        def clean_fields(data: dict) -> dict:
            for key, value in tuple(data.items()):
                if not isinstance(key, str):
                    continue
                if key == "timestamp":
                    data[key] = self._parse_timestamp(value)
            return data

        commits = sorted(self._retrieve_data(self.commits_url), key=lambda x: x["timestamp"])
        return {commit["@id"]: clean_fields(commit) for commit in commits}

    def _download_elements(self):
        elements = self._retrieve_data(self.elements_url)
        max_elements = self.page_size if self.paginate else 100
        if len(elements) == max_elements:
            warn("There are probably more elements that were not retrieved!")
        self._update_elements(elements=elements)

    def _load_from_file(self, file_path: Union[str, Path]):
        self.model = Model.load_from_file(file_path)
