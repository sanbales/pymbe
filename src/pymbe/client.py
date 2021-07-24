import json
import re
from datetime import timezone
from dateutil import parser
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union
from warnings import warn

import ipywidgets as ipyw
import requests
import sysml_v2_api_client as sysml2
import traitlets as trt

from .label import get_label
from .model import Model


TIMEZONES = {
    "CEST": "UTC+2",

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

    _api_configuration: sysml2.Configuration = trt.Instance(sysml2.Configuration)
    _commits_api: sysml2.CommitApi = trt.Instance(sysml2.CommitApi)
    _elements_api: sysml2.ElementApi = trt.Instance(sysml2.ElementApi)
    _projects_api: sysml2.ProjectApi = trt.Instance(sysml2.ProjectApi)

    folder_path: Path = trt.Instance(Path, allow_none=True)
    json_files: Tuple[Path] = ipyw.trait_types.TypedTuple(trt.Instance(Path))
    json_file: Path = trt.Instance(Path, allow_none=True)

    selected_project: str = trt.Unicode(allow_none=True)
    selected_commit: str = trt.Unicode(allow_none=True)

    projects = trt.Dict()

    name_hints = trt.Dict()

    _next_url_regex = re.compile(r'<(http://.*)>; rel="next"')

    @trt.default("_api_configuration")
    def _make_api_configuration(self):
        return sysml2.Configuration(host=self.host)

    @trt.default("_commits_api")
    def _make_commits_api(self):
        with sysml2.ApiClient(self._api_configuration) as client:
            api = sysml2.CommitApi(client)
        return api

    @trt.default("_elements_api")
    def _make_elements_api(self):
        with sysml2.ApiClient(self._api_configuration) as client:
            api = sysml2.ElementApi(client)
        return api

    @trt.default("_projects_api")
    def _make_projects_api(self):
        with sysml2.ApiClient(self._api_configuration) as client:
            # TODO: add check for a bad URL not to make this call
            api = sysml2.ProjectApi(client)
        return api

    @trt.default("projects")
    def _make_projects(self):
        projects = json.loads(self._projects_api.get_projects(_preload_content=False).data)

        def process_project_safely(project) -> dict:
            # protect against projects that can't be parsed
            full_name = project["name"]
            try:
                created = parser.parse(
                    " ".join(full_name.split()[-6:]),
                    tzinfos=TIMEZONES,
                ).astimezone(timezone.utc)
            except ValueError:
                # TODO: revise this when the API server changes the project name
                return dict()
            return dict(
                created=created,
                full_name=full_name,
                name=" ".join(full_name.split()[:-6]),
            )

        results = {
            project["id"]: process_project_safely(project)
            for project in projects
        }

        return {
            project_id: project_data
            for project_id, project_data in results.items()
            if project_data
        }

    @trt.observe("host_url", "host_port")
    def _update_api_configuration(self, *_):
        old_api_configuration = self._api_configuration
        self._api_configuration = self._make_api_configuration()
        if old_api_configuration:
            del old_api_configuration

    @trt.observe("_api_configuration")
    def _update_apis(self, *_):
        for api_type in ("commit", "element", "project"):
            api_attr = f"_{api_type}s_api"
            old_api = getattr(self, api_attr)
            api_maker = getattr(self, f"_make{api_attr}")
            setattr(self, api_attr, api_maker())
            del old_api
        self.projects = self._make_projects()

    @trt.observe("selected_commit")
    def _update_elements(self, *_, elements=None):
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
    def elements_url(self):
        if not self.paginate:
            warn(
                "By default, disabling pagination still retrieves 100 "
                "records at a time!  True pagination is not supported yet."
            )
        return (
            f"{self.host}/"
            f"projects/{self.selected_project}/"
            f"commits/{self.selected_commit}/"
            f"elements"
        ) + f"?page[size]={self.page_size}" if self.page_size else ""

    @lru_cache
    def _retrieve_data(self, url: str) -> List[Dict]:
        """Retrieve model data from a URL using pagination"""
        result = []
        while url:
            response = requests.get(url)

            if not response.ok:
                raise requests.HTTPError(
                    f"Failed to retrieve elements from '{url}', "
                    f"reason: {response.reason}"
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

    def _get_project_commits(self):
        # TODO: add more info about the commit when API provides it
        if not self.selected_project:
            return []
        return [
            commit.id
            for commit in self._commits_api.get_commits_by_project(
                self.selected_project,
            )
        ]

    def _download_elements(self):
        elements = self._retrieve_data(self.elements_url)
        max_elements = self.page_size if self.paginate else 100
        if len(elements) == max_elements:
            warn("There are probably more elements that were not retrieved!")
        self._update_elements(elements=elements)

    def _load_from_file(self, file_path: Union[str, Path]):
        self.model = Model.load_from_file(file_path)
