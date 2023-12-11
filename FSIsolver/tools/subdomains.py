"""
Code snippet from Jørgen Riseth
"""

from typing import Dict, List, Tuple

from dolfin import Mesh, MeshFunction, MeshView
from dolfin.cpp.mesh import MeshFunctionSizet


class FacetView(Mesh):
    def __init__(self, boundaries: MeshFunctionSizet, name: str, value: int):
        super().__init__(MeshView.create(boundaries, value))
        self.rename(name, "")
        self.value = value
        
    def mark_facets(self, subdomain: Mesh, subdomainbdry: MeshFunctionSizet):
        """Label a meshfunction defined on subdomain with the current value."""
        self.build_mapping(subdomain)
        facetmap = self.topology().mapping()[subdomain.id()].cell_map()
        for facet in facetmap: 
            subdomainbdry[facet] = self.value
        return facetmap
            

class SubdomainView(Mesh):
    def __init__(self, subdomains: MeshFunctionSizet, name: str, value: int):
        self.mesh = MeshView.create(subdomains, value)
        super().__init__(self.mesh)
        self.rename(name, "")
        self.value = value
        self.boundaries = MeshFunction('size_t', self.mesh, self.topology().dim()-1, 0)
        
    def mark_boundaries(self, boundarymeshes: List[FacetView]):
        for bdry in boundarymeshes:
            bdry.mark_facets(self, self.boundaries)
        return self.boundaries
            

class SubMeshCollection:
    def __init__(self, subdomains: MeshFunctionSizet, boundaries: MeshFunctionSizet,
                 subdomain_labels: Dict[str, int], boundary_labels: Dict[str, int],
                 subdomain_boundaries: Dict[str, Tuple[str]]):
        self.subdomains = {
            name: SubdomainView(subdomains, name, value) for name, value in subdomain_labels.items()
        }
        self.boundaries = {
            name: FacetView(boundaries, name, value) for name, value in boundary_labels.items()
        }
        self._create_boundary_maps(subdomain_boundaries)
        
    def _create_boundary_maps(self, subdomain_boundaries):
        for subdomain in self.subdomains.values():
            relevant_boundaries = subdomain_boundaries[subdomain.name()]
            subdomain.mark_boundaries([
                self.boundaries[bdry_name] for bdry_name in self.boundaries if bdry_name in relevant_boundaries
            ])