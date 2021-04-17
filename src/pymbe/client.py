from dateutil import parser
from datetime import timezone
from functools import lru_cache
from warnings import warn

import requests
import sysml_v2_api_client as sysml2
import traitlets as trt
import typing as ty

from .core import Base


TIMEZONES = {
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


class SysML2Client(Base):
    """
        A traitleted SysML v2 API Client.

    ..todo:
        - Add ability to use element download pagination.

    """

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

    selected_project: str = trt.Unicode(allow_none=True)
    selected_commit: str = trt.Unicode(allow_none=True)

    projects = trt.Dict()

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
            api = sysml2.ProjectApi(client)
        return api

    @trt.default("projects")
    def _make_projects(self):
        projects = self._projects_api.get_projects()
        # protect against project names that don't parse
        safe_projects = []
        for project in projects:
            try:
                parser.parse(
                    " ".join(project.name.split()[-6:])
                ).astimezone(timezone.utc)
                safe_projects.append(project)
            except ValueError:
                pass

        return {
            project.id: dict(
                created=parser.parse(
                    " ".join(project.name.split()[-6:]),
                    tzinfos=TIMEZONES,
                ).astimezone(timezone.utc),
                full_name=project.name,
                name=" ".join(project.name.split()[:-6]),
            )
            for project in safe_projects
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
        self.relationship_types = sorted({
            element["@type"]
            for element in elements
            if "relatedElement" in element
        })
        self.elements_by_id = {
            element["@id"]: element
            for element in elements
        }

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
        ) + (f"?page[size]={self.page_size}" if self.paginate else "")

    @lru_cache
    def _retrieve_data(self, url: str) -> dict:
        response = requests.get(url)
        if not response.ok:
            raise requests.HTTPError(
                f"Failed to retrieve elements from '{url}', "
                f"reason: {response.reason}"
            )
        return response.json()

    def _get_elements_from_server(self):
        return self._retrieve_data(self.elements_url)

    def update(self, elements: dict):
        elements = tuple(elements.values())
        element_types = {element["@type"] for element in elements}
        self.elements_by_type = {
            element_type: tuple([
                element["@id"]
                for element in elements
                if element["@type"] == element_type
            ])
            for element_type in element_types
        }

    def _download_elements(self):
        elements = self._get_elements_from_server()
        max_elements = self.page_size if self.paginate else 100
        if len(elements) == max_elements:
            warn("There are probably more elements that were not retrieved!")
        self._update_elements(elements=elements)
