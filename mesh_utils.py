import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.utils as pygutils
from torch_sparse import coalesce
from torch_geometric.utils import add_self_loops
import torch
import time

def load_obj(obj_file):
    vertices = []
    faces = []
    try:
        f = open(obj_file)
        for line in f:
            line = line.replace('//', '/')
            if line[:2] == "v ":
                index1 = line.find(" ") + 1
                index2 = line.find(" ", index1 + 1)
                index3 = line.find(" ", index2 + 1)
                vertex = (float(line[index1:index2]), float(
                    line[index2:index3]), float(line[index3:-1]))
                vertices.append(vertex)

            elif line[0] == "f":
                string = line.split(' ')
                string = string[1:]
                face = [int(s.split('/')[0]) -
                        1 for s in string if s.strip() is not '']
                faces.append(face)
        f.close()
    except IOError:
        print(".obj file not found.")

    pos = torch.tensor(vertices, dtype=torch.float)
    face = torch.tensor(faces, dtype=torch.long).t().contiguous()
    data = Data(pos=pos, face=face)
    return data
    
def find_idx(request, target):
    return torch.eq(torch.eq(request, target).sum(1), request.shape[0]).nonzero().item()

def mesh_edge_subdiv(data):
    # face unpooling
    #                          v0
    #                          /\
    #                      n0 /__\ n2
    #                        /\  /\
    #                       /__\/__\
    #                     v1   n1   v2
    # [v0, v1, v2] -> [v0, n0, n2], [n0, v1, n1], [n1, v2, n2], [n0, n1, n2]
    vert = data.pos # V * 3
    V = vert.shape[0]

    # use egde for index (since tensor cannot used for dict keys)
    data.edge_index = None
    data = T.FaceToEdge(remove_faces=False)(data)

    edge_list, _ = data.edge_index.sort(0) # 2 * E
    edge_sort_list, _ = coalesce(edge_list, None, V, V) # 2 * E'
    edge_sort_list = edge_sort_list.t() # E' * 2
    
    face = data.face.t() # F * 3
    F = face.shape[0]

    verts = [face[:, i] for i in range(3)] # [v0 v1 v2]
    edges = [face[:,:2], face[:, 1:], face[:, [-1, 0]]]  # [e0(v0, v1), e1(v1, v2), e2(v2, v0)]
    edges_sort = [edge.sort(1)[0] for edge in edges]
    
    edge_dict = {}
    start_index = vert.shape[0]
    
    new_verts = []
    pool_idx = torch.zeros(1, 2).type_as(edge_list)
    for f in range(F):
        ns = []
        for e in range(3):  
            edge_sort = edges_sort[e][f]
            idx = find_idx(edge_sort, edge_sort_list)
            if idx in edge_dict.keys():
                n = edge_dict[idx]
            else:
                n = start_index
                edge_dict[idx] = torch.tensor([n]).type_as(edge_list)
                pool_idx = torch.cat([pool_idx, edge_sort.unsqueeze(0)], 0)
                start_index += 1        
            ns.append(edge_dict[idx])
        
        if len(new_verts) == 0:
            new_verts.extend(ns)
        else:
            new_verts = [torch.cat([new_verts[i], ns[i]], 0) for i in range(3)]

    pool_idx = pool_idx[1:]  # E" * 2

    v0, v1, v2 = verts
    n0, n1, n2 = new_verts

    new_face0 = torch.stack([v0, n0, n2], 1)
    new_face1 = torch.stack([n0, v1, n1], 1)
    new_face2 = torch.stack([n1, v2, n2], 1)
    new_face3 = torch.stack([n0, n1, n2], 1)
    new_faces = torch.cat([new_face0, new_face1, new_face2, new_face3], 0)  # F" * 3

    return pool_idx, new_faces

def mesh_face_subdiv(data):
    # face unpooling
    #                          v0
    #                          /|\
    #                         / | \
    #                        /n0|  \ 
    #                       /  / \  \
    #                      /__/___\__\
    #                     v1          v2
    # [v0, v1, v2] -> [n0, v0, v1], [n0, v1, v2], [n0, v2, v0]
    
    vert = data.pos # V * 3
    V = vert.shape[0]

    face = data.face.t() # F * 3
    F = face.shape[0]

    verts = [face[:, i] for i in range(3)] # [v0 v1 v2]

    n0 = (torch.arange(F) + V).type_as(face)
    v0, v1, v2 = verts

    new_face0 = torch.stack([n0, v0, v1], 1)
    new_face1 = torch.stack([n0, v1, v2], 1)
    new_face2 = torch.stack([n0, v2, v0], 1)
    new_faces = torch.cat([new_face0, new_face1, new_face2], 0)  # F" * 3

    return face, new_faces

def get_adj_list(data, max_adj=None):
    data.edge_index = None
    data = T.FaceToEdge(remove_faces=False)(data)
    edge, _ = add_self_loops(data.edge_index)
    
    data = T.ToDense()(data)
    adj_mat = data.adj

    num_list = adj_mat.sum(1).long().unsqueeze(1)

    if max_adj is None:
        max_adj = num_list.max().item()
    else:
        max_list = torch.full_like(num_list, max_adj).long()
        num_list = torch.where(num_list > max_adj, max_list, num_list)
    
    adj_list = torch.full_like(adj_mat, -1).long()
    N = data.pos.shape[0]

    for n in range(N):
        adj = adj_mat[n].nonzero()
        num = num_list[n]
        adj_list[n, :num] = adj.t()[:, :num]
    
    adj_list = torch.cat([adj_list[:, :max_adj], num_list], dim=1) # N * max_adj
    return adj_list.type_as(edge), edge

def compute_face_normal(pos, face):
    assert(pos.shape[-1] == face.shape[-1])
    vert = [pos[face[:, i]] for i in range(3)]
    edge1 = vert[1] - vert[0]
    edge2 = vert[2] - vert[0]
    face_normal = torch.cross(edge1, edge2)
    return face_normal

def compute_vert_normal(pos, face):
    assert(pos.shape[1] == face.shape[1])
    face_normal = compute_face_normal(pos, face)
    vert_id = [face[:, i] for i in range(3)]

    normalize = F.normalize
    unit_normal = normalize(face_normal, dim=1)
    vert_normal = torch.zeros_like(pos)
    for v in vert_id:
        vert_normal = vert_normal.index_add(0, v, unit_normal)
    vert_normal = normalize(vert_normal, dim=1)
    return vert_normal

def reparam_sample(pos, face, max_num=5000):
    dist_uni = torch.distributions.Uniform(
        torch.tensor([0.0]), torch.tensor([1.0]))

    face_normal = compute_face_normal(pos, face)

    normalize = F.normalize
    unit_normal = normalize(face_normal, p=2)

    areas = face_normal.norm(dim=1)
    areas_ratio = areas / areas.sum()

    sample = torch.multinomial(areas_ratio, max_num, replacement=True)
    sample_faces = face[sample]
    sample_normals = unit_normal[sample]

    u1 = torch.sqrt(dist_uni.sample((max_num, ))).to(face.device)
    u2 = dist_uni.sample((max_num, )).to(face.device)
    
    vs = [torch.index_select(pos, 0, sample_faces[:, i]) for i in range(3)]
    ps = [1-u1, u1 * (1-u2), u1*u2]
    sample_points = ps[0] * vs[0] + ps[1] * vs[1] + ps[2] * vs[2]
    
    return sample_points, sample_normals

# construct dual mesh 
def dual_mesh(vert_feats, face):
    assert(len(vert_feats.shape) == 2)
    assert(len(face.shape) == 2)
    assert(face.shape[-1] == 3)

    edges = [face[:,:2], face[:, 1:], face[:, [-1, 0]]]  # [e0(v0, v1), e1(v1, v2), e2(v2, v0)]
    edges_sort = [edge.sort(1)[0] for edge in edges]
    
    edge_dict = {}
    
    F = face.shape[0]
    for f in range(F):
        for e in range(3):
            edge = tuple(edges_sort[e][f].tolist())
            if edge in edge_dict.keys():
                edge_dict[edge].append(f)
            else:
                edge_dict[edge] = [f]

    vert_feat = [vert_feats[face[:, i]] for i in range(3)]
    face_feats = (vert_feat[0] + vert_feat[1] + vert_feat[2]) / 3
    
    dual_edge_lists = []
    for k, v in edge_dict.items():
        edge = torch.tensor([v[0], v[1]]).long()
        dual_edge_lists.append(edge)
        edge_inv = torch.tensor([v[1], v[0]]).long()
        dual_edge_lists.append(edge_inv)
    
    dual_edge = torch.stack(dual_edge_lists, dim=0).t().to(face_feats.device)

    return face_feats, dual_edge

# construct line mesh 
def line_mesh_slow(vert_feats, face):
    assert(len(vert_feats.shape) == 2)
    assert(len(face.shape) == 2)
    assert(face.shape[-1] == 3)
    
    edges = [face[:,:2], face[:, 1:], face[:, [-1, 0]]]  # [e0(v0, v1), e1(v1, v2), e2(v2, v0)]
    edges_sort = [edge.sort(1)[0] for edge in edges]
    
    edge_dict = {}
    edge_node = []
    F = face.shape[0]
    idx = 0
    for f in range(F):
        edge0 = tuple(edges_sort[0][f].tolist())
        if edge0 in edge_dict.keys():
            idx0 = edge_dict[edge0][0]
        else:
            idx0 = idx
            edge_dict[edge0] = [idx0]
            edge_node.append(edges_sort[0][f])
            idx += 1

        edge1 = tuple(edges_sort[1][f].tolist())
        if edge1 in edge_dict.keys():
            idx1 = edge_dict[edge1][0]
        else:
            idx1 = idx
            edge_dict[edge1] = [idx1]
            edge_node.append(edges_sort[1][f])
            idx += 1

        edge2 = tuple(edges_sort[2][f].tolist())
        if edge2 in edge_dict.keys():
            idx2 = edge_dict[edge2][0]
        else:
            idx2 = idx
            edge_dict[edge2] = [idx2]
            edge_node.append(edges_sort[2][f])
            idx += 1
        
        edge_dict[edge0].extend([idx1, idx2])
        edge_dict[edge1].extend([idx0, idx2])
        edge_dict[edge2].extend([idx0, idx1])
    
    edge_node = torch.stack(edge_node, 0)
    vert_feat = [vert_feats[edge_node[:, i]] for i in range(2)]
    line_feats = (vert_feat[0] + vert_feat[1]) / 2
    
    line_edge_lists = []
    for k, v in edge_dict.items():
        for i in range(4):
            edge = torch.tensor([v[0], v[i+1]]).long()
            line_edge_lists.append(edge)
    
    line_edge = torch.stack(line_edge_lists, dim=0).t().to(line_feats.device)
    
    return line_feats, line_edge, edge_node

def line_mesh(vert_feats, face):
    assert(len(vert_feats.shape) == 2)
    assert(len(face.shape) == 2)
    assert(face.shape[-1] == 3)
    
    V = vert_feats.shape[0]
    F = face.shape[0]
    edges = torch.cat([face[:,:2], face[:, 1:], face[:, [-1, 0]]], 0).sort(1)[0]
    value = edges[:, 0] * V + edges[:, 1]
    edge_feats = 0.5 * (vert_feats[edges[:, 0]] + vert_feats[edges[:, 1]])
    uniq, inv = torch.unique(value, sorted=True, return_inverse=True)
    edge_keys = torch.split(inv, F)

    line_idx = torch.arange(uniq.shape[0]).to(vert_feats.device)
    line_edge = torch.cat([
                    torch.stack([line_idx[edge_keys[0]], line_idx[edge_keys[1]]], -1), 
                    torch.stack([line_idx[edge_keys[1]], line_idx[edge_keys[0]]], -1),
                    torch.stack([line_idx[edge_keys[0]], line_idx[edge_keys[2]]], -1),
                    torch.stack([line_idx[edge_keys[2]], line_idx[edge_keys[0]]], -1), 
                    torch.stack([line_idx[edge_keys[1]], line_idx[edge_keys[2]]], -1),
                    torch.stack([line_idx[edge_keys[2]], line_idx[edge_keys[1]]], -1),
                    ], 0).t()

    line_feats = torch.zeros(uniq.shape[0], edge_feats.shape[-1]).type_as(vert_feats)
    line_feats.index_copy_(0, inv, edge_feats)

    edge_node = torch.zeros(uniq.shape[0], edges.shape[-1]).type_as(face)
    edge_node.index_copy_(0, inv, edges)

    return line_feats, line_edge, edge_node


def symmetry_edge(data):
    verts = data.pos
    verts_dict = {tuple(verts[i].tolist()):i for i in range(verts.shape[0])}
    
    symm_edges = []
    for src in verts_dict.keys():
        x, y, z = src
        tar = (-x, y, z)
        if tar in verts_dict.keys():
            symm_edges.append(torch.tensor([verts_dict[src], verts_dict[tar]]))

    symm_edges = torch.stack(symm_edges, -1).to(verts.device)
    return symm_edges