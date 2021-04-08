import ipywidgets as ipyw
import traitlets as trt
import typing as ty

from ..core import Base


@ipyw.register
class BaseWidget(Base, ipyw.DOMWidget):
    """A base widget to enforce standardization with selectors."""

    description = trt.Unicode("Unnamed Widget").tag(sync=True)
    selected: ty.Tuple[str] = trt.Tuple()

    def update_selected(self, *new_selections):
        if set(self.selected).symmetric_difference(new_selections):
            self.selected = new_selections
