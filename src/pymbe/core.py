import traitlets as trt
import typing as ty
import warnings


class Base(trt.HasTraits):
    """A common base class to standardize properties and interactions"""

    elements_by_id: ty.Dict[str, dict] = trt.Dict(
        key_trait=trt.Unicode(),
        value_trait=trt.Dict(),
    )

    elements_by_type: ty.Dict[str, ty.Tuple[str]] = trt.Dict(
        key_trait=trt.Unicode(),
        value_trait=trt.Tuple(),
    )

    relationship_types: ty.Tuple[str] = trt.Tuple()

    def update(self, elements: dict):
        """Subclasses must implement this!"""
        warnings.warn(
            f"{self.__class__.__name__} does not "
            f"specialize the `update` method!"
        )

    def get_element_by_id(self, id_: str) -> dict:
        return self.elements_by_id[id_]

    def get_name_by_id(self, id_: str) -> str:
        return self.get_element_by_id(id_).get("name")

    @trt.observe("elements_by_id")
    def _update_elements(self, *_):
        self.update(elements=self.elements_by_id)
