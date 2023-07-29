import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from warnings import warn

import ipywidgets as ipyw
import requests
import traitlets as trt
from dateutil import parser

from .model import Model, ModelClient

URL_CACHE_SIZE = 1024

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


@dataclass
class Item:
    _id: str
    created: Optional[Union[str, datetime]] = None
    description: Optional[str] = None

    @classmethod
    def all_subclasses(cls) -> Dict[str, Type]:
        return {
            klass.__name__: klass
            for klass in (
                set(cls.__subclasses__()).union(
                    [s for c in cls.__subclasses__() for s in c.all_subclasses()]
                )
            )
        }

    @classmethod
    def parse(cls, data: dict) -> "Item":
        def dereference(value):
            if isinstance(value, (list, set, tuple)):
                return tuple(dereference(a_value) for a_value in value)
            if isinstance(value, dict) and len(value) == 1 and "@id" in value:
                return value["@id"]
            return value

        data = {key.replace("@", ""): dereference(value) for key, value in data.items()}
        # get the dataclass by the type
        cls = cls.all_subclasses().get(data.pop("type"))
        if not cls:
            return None
        return cls(**data)


@dataclass
class ProjectItem(Item):
    owning_project: Optional["Project"] = None


@dataclass
class Commit(ProjectItem):
    previous_commit: Optional["Commit"] = None


@dataclass
class Branch(ProjectItem):
    head: Optional["Commit"] = None
    referenced_commit: Optional["Commit"] = None


@dataclass
class Project(Item):
    tags: Optional[Tuple[Commit]] = None
    branches: Optional[Tuple[Branch]] = None
    commits: Optional[Tuple[Commit]] = None


class APIClient(trt.HasTraits, ModelClient):
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
        default_value=200,
        min=1,
    )

    paginate = trt.Bool(default_value=True)

    selected_project: str = trt.Unicode(allow_none=True)
    selected_commit: str = trt.Unicode(allow_none=True)

    projects = trt.Dict()

    name_hints = trt.Dict()

    log_out: ipyw.Output = trt.Instance(ipyw.Output, args=())

    _next_url_regex = re.compile(r'<(http://.*)>; rel="next"')

    @trt.default("projects")
    def _make_projects(self):
        def process_project_safely(project) -> Dict[str, str]:
            # protect against projects that can't be parsed
            try:
                name_with_date = project["name"] or f"""Project(id={project["@id"]})"""
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

    @property
    def host(self):
        return f"{self.host_url}:{self.host_port}"

    @property
    def projects_url(self):
        return f"{self.host}/projects"

    @property
    def branches_url(self):
        return f"{self.projects_url}/{self.selected_project}/branches"

    @property
    def tags_url(self):
        return f"{self.projects_url}/{self.selected_project}/tags"

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
        return f"{self.commits_url}/{self.selected_commit}/elements"

    def reset_cache(self):
        self._retrieve_paginated_data.cache_clear()

    @staticmethod
    def _retrieve_data(
        url: str, process_response: bool = True
    ) -> Union[List, Dict, requests.Response]:
        response = requests.get(url)
        if not response.ok:
            raise requests.HTTPError(
                f"Failed to retrieve elements from '{url}', reason: {response.reason}"
            )
        if not process_response:
            return response
        return response.json()

    @lru_cache(maxsize=URL_CACHE_SIZE)
    def _retrieve_paginated_data(
        self, url: str, on_page: Callable = None, remove_empty_data: bool = True
    ) -> List[Dict]:
        """Retrieve model data from a URL using pagination"""
        data = []
        while url:
            response = self._retrieve_data(url, process_response=False)
            data += response.json()
            if on_page:
                on_page()
            link = response.headers.get("Link")
            if not link:
                break
            urls = self._next_url_regex.findall(link)
            if len(urls) > 1:
                raise requests.HTTPError(
                    "Found multiple 'next' pagination urls: "
                    ", ".join(map(lambda x: f"<{x}>", urls))
                )
            url = urls[0] if urls else None
        if remove_empty_data:
            return [item for item in data if item]
        return data

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
                if key == "created":
                    data[key] = self._parse_timestamp(value)
            return data

        commits = sorted(self._retrieve_data(self.commits_url), key=lambda x: x["created"])
        return {commit["@id"]: clean_fields(commit) for commit in commits}

    def get_model(self, on_page: Callable = None) -> Optional[Model]:
        """Download a model from the current `elements_url`."""
        if not self.selected_commit:
            return None
        url = self.elements_url
        if self.page_size:
            url += f"?page[size]={self.page_size}"
        elements = self._retrieve_paginated_data(url=url, on_page=on_page)
        total = len(elements)
        # Remove bogus empty elements
        elements = [
            element for element in elements if isinstance(element, dict) and "@id" in element
        ]
        if total > len(elements):
            warn(f"Downloaded {total-len(elements)} bad elements, will have to ignore them")
        if not elements:
            return None

        max_elements = self.page_size if self.paginate else 100
        if len(elements) == max_elements:
            warn("There are probably more elements that were not retrieved!")

        return Model.load(
            elements=elements,
            name=self.projects[self.selected_project]["name"],
            source=self.elements_url,
            _api=self,
        )

    def get_element_data(self, element_id: str) -> dict:
        try:
            return self._retrieve_data(f"{self.elements_url}/{element_id}")
        except requests.HTTPError:
            warn(f"Could not retrieve data for {element_id}")
            return {"@id": element_id, "@type": "Element"}
