from copy import deepcopy
from dateutil import parser
from datetime import timezone
from functools import lru_cache
from warnings import warn

import requests
import sysml_v2_api_client as sysml2
import traitlets as trt

from .core import Base
from .label import get_label


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

    contexts: dict = trt.Dict(kw={})

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

        def process_project_safely(project) -> dict:
            # protect against projects that can't be parsed
            try:
                created = parser.parse(
                    " ".join(project.name.split()[-6:]),
                    tzinfos=TIMEZONES,
                ).astimezone(timezone.utc)
            except ValueError:
                # TODO: revise this when the API server changes the project name
                return dict()
            return dict(
                created=created,
                full_name=project.name,
                name=" ".join(project.name.split()[:-6]),
            )

        results = {
            project.id: process_project_safely(project)
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
        elements_by_id = {
            element["@id"]: self.import_context(element)
            for element in elements
        }
        for element in elements_by_id.values():
            if "label" not in element:
                element["label"] = get_label(
                    element,
                    all_elements=elements_by_id,
                )
        self.elements_by_id = elements_by_id

    @property
    def host(self):
        if not self.host_port:
            return self.host_url
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

    def get_context_url(self, metatype):
        return f"{self.host}/jsonld/{metatype}.jsonld"

    @lru_cache
    def _retrieve_data(self, url: str) -> dict:
        response = requests.get(url)
        if not response.ok:
            raise requests.HTTPError(
                f"Failed to retrieve elements from '{url}', "
                f"reason: {response.reason}"
            )
        return response.json()

    def import_context(self, jsonld_item: dict) -> dict:
        jsonld_item = deepcopy(jsonld_item)
        metatype = jsonld_item.get("@type")
        if not metatype:
            return jsonld_item
        context = jsonld_item.get("@context", {})
        if not context:
            context_url = self.get_context_url(metatype)
        else:
            context_url = context.get("@import", None)
        if not context_url:
            return jsonld_item
        jsonld_item["@context"] = context
        if context_url not in self._cached_contexts:
            response = requests.get(context_url)
            if not response.ok:
                raise requests.HTTPError(response.reason)
            data = response.json()
            if "@context" not in data:
                raise ValueError(
                    "Download context does not have a "
                    f"@context key: {list(data.keys())}"
                )
            self._cached_contexts[context_url] = data["@context"]
        jsonld_item["@context"].update(self._cached_contexts[context_url])
        return jsonld_item

    def _get_elements_from_server(self):
        return self._retrieve_data(self.elements_url)

    def update(self, elements: dict):
        """All the functionality for the update is already handled"""
        pass

    def _download_elements(self):
        elements = self._get_elements_from_server()
        max_elements = self.page_size if self.paginate else 100
        if len(elements) == max_elements:
            warn("There are probably more elements that were not retrieved!")
        self._update_elements(elements=elements)
