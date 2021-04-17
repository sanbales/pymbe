import ipywidgets as ipyw
import traitlets as trt
import typing as ty

from .core import BaseWidget


@ipyw.register
class ElementInspector(ipyw.Output, BaseWidget):
    """A widget to inspect a SysML v2 Element"""

    FILTER_KEYS = ("@context",)

    description = trt.Unicode("Inspector").tag(sync=True)

    include_empty = trt.Bool(default_value=False)

    clean_data: ty.Dict[str, dict] = trt.Dict(
        key_trait=trt.Unicode(),
        value_trait=trt.Dict(),
    )

    @trt.validate("layout")
    def _validate_layout(self, proposal):
        layout = proposal.value
        layout.overflow_y = "auto"
        layout.width = "auto"
        return layout

    @trt.observe("selected")
    def _update_details(self, *_):
        self.outputs = self._make_json_output()

    @trt.observe("include_empty")
    def _update_data(self):
        self.update(self.elements_by_id)

    def get_clean_data(self, *,
        element_id: str = None,
        data: dict = None,
    ) -> dict:
        data = data or self.elements_by_id.get(element_id, {})
        return {
            key: value
            for key, value in data.items()
            if key not in self.FILTER_KEYS
            and (
                self.include_empty
                or value
                or value is False
                or value == 0.0
            )
        }

    def update(self, elements: dict):
        self.clean_data = {
            id_: self.get_clean_data(data=data)
            for id_, data in elements.items()
        }

    @staticmethod
    def _get_name(data: dict) -> str:
        for key in ("name", "qualifiedName"):
            name = data.get(key, None)
            if name:
                return str(name)
        return f"""«{data["@type"]}: {data["@id"]}»"""


    def _make_json_output(self) -> list:
        data = {
            id_: self.clean_data.get(id_)
            for id_ in self.selected
            if id_ in self.clean_data
        }
        names = {
            id_: self._get_name(data_)
            for id_, data_ in data.items()
        }
        return [
            {
                "output_type": "display_data",
                "data": {
                    "text/plain": f"JSON Display for {id_}",
                    "application/json": data[id_],
                },
                "metadata": {
                    "application/json": {
                        "expanded": False,
                        "root": names[id_],
                    },
                },
            }
            for id_ in self.selected
        ]
