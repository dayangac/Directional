"""Port of tutorial/301_PowerFields: click on the mesh to constrain the nearest vertex."""
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import directional as dr

N = 4
DATA = Path(__file__).parents[2] / "tutorial" / "data"

mesh = dr.readOBJ(str(DATA / "rocker-arm2500.obj"))
vtb = dr.IntrinsicVertexTangentBundle(mesh)
const_vertices, const_vectors = [], []
normalize_field, view_hard = False, 0


def update():
    weight = -1.0 if view_hard == 0 else 10e-4
    power = dr.power_field(vtb, np.array(const_vertices, dtype=int), np.array(const_vectors).reshape(-1, 3),
                           np.full(len(const_vertices), weight), N, normalize_field)
    raw = dr.power_to_raw(power, N)
    dr.principal_matching(raw)

    for i in range(N):
        ps_mesh.add_vector_quantity(f"Vector {i}", raw.extField[:, 3 * i:3 * i + 3], enabled=True)
    sings = ps.register_point_cloud("Singularities", mesh.barycenters[raw.singLocalCycles])
    sings.add_scalar_quantity("index", raw.singIndices / N, enabled=True, cmap="coolwarm", vminmax=(-1, 1))
    if const_vertices:
        constraints = ps.register_point_cloud("Constraints", mesh.V[const_vertices], color=(1, 1, 0))
        constraints.add_vector_quantity("direction", np.array(const_vectors), enabled=True, color=(1, 1, 0))


def callback():
    global normalize_field, view_hard
    changed, normalize_field = psim.Checkbox("Normalize Field", normalize_field)
    combo_changed, view_hard = psim.Combo("Viewing mode", view_hard, ["Hard prescription", "Soft prescription"])
    if changed or combo_changed:
        update()

    if psim.IsMouseClicked(0) and not psim.GetIO().WantCaptureMouse:
        pick = ps.pick(screen_coords=psim.GetMousePos())
        if pick.is_hit and pick.structure_name == "mesh" and "bary_coords" in pick.structure_data:
            face = mesh.F[pick.structure_data["index"]]
            vertex = int(face[np.argmax(pick.structure_data["bary_coords"])])
            vector = pick.position - mesh.V[vertex]
            vector /= np.linalg.norm(vector)
            if vertex in const_vertices:
                const_vectors[const_vertices.index(vertex)] = vector
            else:
                const_vertices.append(vertex)
                const_vectors.append(vector)
            update()


ps.init()
ps_mesh = ps.register_surface_mesh("mesh", mesh.V, mesh.F)
ps_mesh.set_selection_mode("faces_only")
update()
ps.set_user_callback(callback)
ps.show()
