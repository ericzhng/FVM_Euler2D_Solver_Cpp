#!/usr/bin/env python3
"""
Mesh Partitioner for FVM2D C++ Solver

This script partitions a mesh into N parts using METIS and writes
partition files in the simple ASCII format expected by the C++ solver.

Usage:
    python partition_mesh.py <input_mesh.msh> <num_partitions> <output_dir>

Example:
    python partition_mesh.py ../data/mesh.msh 4 mesh/
"""

import sys
import os
import numpy as np

try:
    import meshio
except ImportError:
    print("Error: meshio is required. Install with: pip install meshio")
    sys.exit(1)

try:
    import pymetis
except ImportError:
    print("Warning: pymetis not found. Install with: pip install pymetis")
    print("Falling back to simple partitioning...")
    pymetis = None


def compute_cell_centroid(nodes, node_coords):
    """Compute centroid of a cell."""
    coords = node_coords[nodes]
    return np.mean(coords, axis=0)


def compute_cell_volume(nodes, node_coords):
    """Compute area of a 2D cell using shoelace formula."""
    coords = node_coords[nodes]
    n = len(nodes)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i, 0] * coords[j, 1] - coords[j, 0] * coords[i, 1]
    return abs(area) * 0.5


def compute_face_data(nodes, node_coords, centroid):
    """Compute face normals, midpoints, and areas for a cell."""
    n = len(nodes)
    face_data = []

    for i in range(n):
        j = (i + 1) % n
        p1 = node_coords[nodes[i]]
        p2 = node_coords[nodes[j]]

        # Face midpoint
        midpoint = 0.5 * (p1 + p2)

        # Face length (area in 2D)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        area = np.sqrt(dx*dx + dy*dy)

        # Outward normal (rotate edge by 90 degrees)
        # Normal should point away from cell centroid
        nx, ny = dy, -dx
        normal_len = np.sqrt(nx*nx + ny*ny)
        if normal_len > 1e-12:
            nx /= normal_len
            ny /= normal_len

        # Ensure normal points outward
        to_face = midpoint - centroid
        if nx * to_face[0] + ny * to_face[1] < 0:
            nx, ny = -nx, -ny

        face_data.append({
            'nodes': (nodes[i], nodes[j]),
            'midpoint': midpoint,
            'normal': np.array([nx, ny]),
            'area': area
        })

    return face_data


def build_cell_neighbors(cells, face_to_cells):
    """Build cell neighbor connectivity."""
    num_cells = len(cells)
    neighbors = [[] for _ in range(num_cells)]

    for face, cell_list in face_to_cells.items():
        if len(cell_list) == 2:
            c1, c2 = cell_list
            neighbors[c1].append(c2)
            neighbors[c2].append(c1)

    return neighbors


def partition_mesh_metis(adjacency, num_parts):
    """Partition mesh using METIS."""
    if pymetis is None:
        return partition_mesh_simple(len(adjacency), num_parts)

    # Convert to METIS format (CSR-like)
    xadj = [0]
    adjncy = []
    for neighbors in adjacency:
        adjncy.extend(neighbors)
        xadj.append(len(adjncy))

    _, membership = pymetis.part_graph(num_parts, adjacency=adjacency)
    return membership


def partition_mesh_simple(num_cells, num_parts):
    """Simple partitioning by cell index."""
    cells_per_part = num_cells // num_parts
    membership = []
    for i in range(num_cells):
        membership.append(min(i // cells_per_part, num_parts - 1))
    return membership


def write_partition_file(filepath, rank, num_ranks, mesh_data):
    """Write a partition file in the simple ASCII format."""
    with open(filepath, 'w') as f:
        f.write(f"# FVM2D Partition Mesh Format v1.0\n")
        f.write(f"# Partition: {rank} of {num_ranks}\n\n")

        # HEADER
        f.write("HEADER\n")
        f.write(f"num_nodes: {mesh_data['num_nodes']}\n")
        f.write(f"num_owned_cells: {mesh_data['num_owned']}\n")
        f.write(f"num_halo_cells: {mesh_data['num_halo']}\n")
        f.write(f"num_boundary_patches: {len(mesh_data['boundary_patches'])}\n\n")

        # BOUNDARY_PATCHES
        f.write("BOUNDARY_PATCHES\n")
        for tag, name in mesh_data['boundary_patches'].items():
            f.write(f"{tag} {name}\n")
        f.write("\n")

        # NODES
        f.write("NODES\n")
        for i, coord in enumerate(mesh_data['node_coords']):
            f.write(f"{i} {coord[0]:.12e} {coord[1]:.12e}\n")
        f.write("\n")

        # CELLS
        f.write("CELLS\n")
        for cell in mesh_data['cells']:
            line = f"{cell['local_id']} {cell['global_id']} {len(cell['nodes'])}"
            for n in cell['nodes']:
                line += f" {n}"
            line += f" {len(cell['faces'])}"
            for face in cell['faces']:
                line += f" {face['neighbor']} {face['tag']}"
                line += f" {face['normal'][0]:.12e} {face['normal'][1]:.12e}"
                line += f" {face['midpoint'][0]:.12e} {face['midpoint'][1]:.12e}"
                line += f" {face['area']:.12e}"
            f.write(line + "\n")
        f.write("\n")

        # SEND_MAP
        f.write("SEND_MAP\n")
        for neighbor_rank, indices in mesh_data['send_map'].items():
            f.write(f"{neighbor_rank} {len(indices)}")
            for idx in indices:
                f.write(f" {idx}")
            f.write("\n")
        f.write("\n")

        # RECV_MAP
        f.write("RECV_MAP\n")
        for neighbor_rank, indices in mesh_data['recv_map'].items():
            f.write(f"{neighbor_rank} {len(indices)}")
            for idx in indices:
                f.write(f" {idx}")
            f.write("\n")
        f.write("\n")

        f.write("END\n")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    num_parts = int(sys.argv[2])
    output_dir = sys.argv[3]

    print(f"Reading mesh from: {input_file}")
    mesh = meshio.read(input_file)

    # Extract nodes
    node_coords = mesh.points[:, :2]  # 2D only
    num_nodes = len(node_coords)
    print(f"  Nodes: {num_nodes}")

    # Extract cells (triangles and quads)
    cells = []
    for cell_block in mesh.cells:
        if cell_block.type in ['triangle', 'quad']:
            for cell_nodes in cell_block.data:
                cells.append(list(cell_nodes))

    num_cells = len(cells)
    print(f"  Cells: {num_cells}")

    # Build face-to-cell map
    face_to_cells = {}
    for cell_idx, cell_nodes in enumerate(cells):
        n = len(cell_nodes)
        for i in range(n):
            j = (i + 1) % n
            face = tuple(sorted([cell_nodes[i], cell_nodes[j]]))
            if face not in face_to_cells:
                face_to_cells[face] = []
            face_to_cells[face].append(cell_idx)

    # Build adjacency
    adjacency = build_cell_neighbors(cells, face_to_cells)

    # Identify boundary faces and patches
    boundary_faces = {}
    boundary_tag = 1
    for face, cell_list in face_to_cells.items():
        if len(cell_list) == 1:
            boundary_faces[face] = boundary_tag

    # Simple boundary patches (could be extended to read from mesh)
    boundary_patches = {1: "boundary"}

    # Partition
    print(f"Partitioning into {num_parts} parts...")
    membership = partition_mesh_metis(adjacency, num_parts)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each partition
    for rank in range(num_parts):
        print(f"  Processing partition {rank}...")

        # Find cells owned by this rank
        owned_cells = [i for i, m in enumerate(membership) if m == rank]
        owned_set = set(owned_cells)

        # Find halo cells (neighbors of owned cells in other partitions)
        halo_cells = set()
        for cell_idx in owned_cells:
            for neighbor in adjacency[cell_idx]:
                if neighbor not in owned_set:
                    halo_cells.add(neighbor)
        halo_cells = list(halo_cells)

        # Build send/recv maps
        send_map = {}  # rank -> local indices to send
        recv_map = {}  # rank -> local indices to receive

        # Global to local mapping
        g2l = {}
        for local_idx, global_idx in enumerate(owned_cells):
            g2l[global_idx] = local_idx
        for local_idx, global_idx in enumerate(halo_cells, start=len(owned_cells)):
            g2l[global_idx] = local_idx

        # Build communication maps
        for cell_idx in owned_cells:
            for neighbor in adjacency[cell_idx]:
                if neighbor not in owned_set:
                    neighbor_rank = membership[neighbor]
                    if neighbor_rank not in send_map:
                        send_map[neighbor_rank] = []
                    local_idx = g2l[cell_idx]
                    if local_idx not in send_map[neighbor_rank]:
                        send_map[neighbor_rank].append(local_idx)

        for halo_idx in halo_cells:
            halo_rank = membership[halo_idx]
            if halo_rank not in recv_map:
                recv_map[halo_rank] = []
            recv_map[halo_rank].append(g2l[halo_idx])

        # Collect nodes used by this partition
        used_nodes = set()
        for global_idx in owned_cells + halo_cells:
            used_nodes.update(cells[global_idx])
        used_nodes = sorted(used_nodes)

        # Node global to local mapping
        node_g2l = {g: l for l, g in enumerate(used_nodes)}
        local_node_coords = node_coords[used_nodes]

        # Build cell data
        cell_data = []
        all_cells = owned_cells + halo_cells

        for local_idx, global_idx in enumerate(all_cells):
            cell_nodes = cells[global_idx]
            local_nodes = [node_g2l[n] for n in cell_nodes]
            centroid = compute_cell_centroid(local_nodes, local_node_coords)

            # Compute face data
            face_info = []
            for i in range(len(cell_nodes)):
                j = (i + 1) % len(cell_nodes)
                face_key = tuple(sorted([cell_nodes[i], cell_nodes[j]]))

                p1 = local_node_coords[node_g2l[cell_nodes[i]]]
                p2 = local_node_coords[node_g2l[cell_nodes[j]]]
                midpoint = 0.5 * (p1 + p2)

                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                area = np.sqrt(dx*dx + dy*dy)
                nx, ny = dy / area if area > 1e-12 else 0, -dx / area if area > 1e-12 else 0

                # Ensure outward normal
                to_face = midpoint - centroid
                if nx * to_face[0] + ny * to_face[1] < 0:
                    nx, ny = -nx, -ny

                # Find neighbor
                face_cells = face_to_cells[face_key]
                if len(face_cells) == 2:
                    neighbor_global = face_cells[0] if face_cells[1] == global_idx else face_cells[1]
                    if neighbor_global in g2l:
                        neighbor_local = g2l[neighbor_global]
                    else:
                        neighbor_local = -1  # Not in this partition
                    tag = 0
                else:
                    neighbor_local = -1
                    tag = boundary_faces.get(face_key, 1)

                face_info.append({
                    'neighbor': neighbor_local,
                    'tag': tag,
                    'normal': np.array([nx, ny]),
                    'midpoint': midpoint,
                    'area': area
                })

            cell_data.append({
                'local_id': local_idx,
                'global_id': global_idx,
                'nodes': local_nodes,
                'faces': face_info
            })

        # Write partition file
        mesh_data = {
            'num_nodes': len(local_node_coords),
            'num_owned': len(owned_cells),
            'num_halo': len(halo_cells),
            'boundary_patches': boundary_patches,
            'node_coords': local_node_coords,
            'cells': cell_data,
            'send_map': send_map,
            'recv_map': recv_map
        }

        output_file = os.path.join(output_dir, f"partition_{rank}.mesh")
        write_partition_file(output_file, rank, num_parts, mesh_data)
        print(f"    Written: {output_file}")
        print(f"    Owned: {len(owned_cells)}, Halo: {len(halo_cells)}")

    print("Done!")


if __name__ == "__main__":
    main()
