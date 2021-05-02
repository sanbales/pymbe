import json

import rdflib as rdf
import traitlets as trt

from ..core import Base


class SysML2RDFGraph(Base):
    """A Resource Description Framework (RDF) Graph for SysML v2 data."""

    graph: rdf.Graph = trt.Instance(rdf.Graph, args=tuple())
    merge: bool = trt.Bool(default_value=False)

    def __repr__(self):
        return (
            "<SysML v2 RDF Graph: "
            f"{len(self.graph):,d} triples, "
            f"{len(set(self.graph.predicates())):,d} unique predicates"
            ">"
        )

    def update(self, elements: dict):
        if not self.merge:
            old_graph = self.graph
            self.graph = rdf.Graph()
            del old_graph

        self.graph.parse(
            data=json.dumps(list(elements.values())),
            format="application/ld+json",
        )
