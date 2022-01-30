import typing as ty

import ipywidgets as ipyw
import traitlets as trt

from ..model import Model


@ipyw.register
class BaseWidget(ipyw.DOMWidget):
    """A base widget to enforce standardization with selectors."""

    closable: bool = trt.Bool(False).tag(sync=True)
    description = trt.Unicode("Unnamed Widget").tag(sync=True)

    model: Model = trt.Instance(Model, allow_none=True, help="The instance of the SysML model.")

    selected: ty.Tuple[str] = ipyw.widgets.trait_types.TypedTuple(
        trait=trt.Unicode(),
    ).tag(sync=True)

    log_out: ipyw.Output = trt.Instance(ipyw.Output, args=())

    @trt.validate("description")
    def _customize_description(self, proposal: trt.Bunch):
        description = proposal.value
        if description == "Unnamed Widget":
            description = self.__class__.__name__
        return description

    @trt.observe("model")
    def _update(self, change: trt.Bunch):
        with self.log_out:
            self.update(change)

    def update(self, change: trt.Bunch):
        raise NotImplementedError("Each widget needs to specify their own update method!")

    def update_selected(self, *new_selections: str):
        if set(self.selected).symmetric_difference(new_selections):
            self.selected = new_selections
