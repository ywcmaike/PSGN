import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .chamfer import ChamferDistance
import time
import sys
sys.path.append('.')
from mesh_utils import *

def laplace_coord(coord, lap_idx):
    vertex = torch.cat([coord, torch.zeros([1, 3]).type_as(coord)], 0)
    indices = lap_idx[:, :-1]

    weights = lap_idx[:, -1].float()
    weights = torch.reciprocal(weights).view([-1, 1]).repeat(1, 3)

    laplace = torch.sum(vertex[indices], 1)
    laplace = coord - torch.mul(laplace, weights)
    return laplace

def compute_laplace_loss(pre_coord, coord, lap_idx):
    lap1 = laplace_coord(pre_coord, lap_idx)
    lap2 = laplace_coord(coord, lap_idx)
    laplace_loss = torch.mean(torch.sum((lap1 - lap2) ** 2, 1))
    return laplace_loss

def compute_laplace_loss_new(coord, lap_idx):
    lap = laplace_coord(coord, lap_idx)
    laplace_loss = torch.mean(torch.sum(lap ** 2, 1))
    return laplace_loss

def compute_laplace_loss_new_sub(coord, edge):
    adj_mat = pygutils.to_dense_adj(edge).squeeze(0)

    adj_mat = adj_mat - torch.eye(adj_mat.shape[0]).type_as(adj_mat)
    adj_inv_sum = torch.reciprocal(adj_mat.sum(1)).float()
    weight_mat = adj_mat.float() * adj_inv_sum.unsqueeze(-1).expand(-1, adj_mat.shape[1])
    weight_mat = weight_mat - torch.eye(adj_mat.shape[0]).type_as(weight_mat)
    assert((weight_mat >= -1).all()) 
    laplace_loss = torch.mean(torch.sum((weight_mat.mm(coord)) ** 2, 1))
    return laplace_loss
    
def compute_convex_loss(coord, face, edge):
    vertex_normal = compute_vert_normal(coord, face)
    
    normalize = F.normalize

    coord0 = coord[edge[:, 0]]
    coord1 = coord[edge[:, 1]]
    normal0 = vertex_normal[edge[:, 0]]
    normal1 = vertex_normal[edge[:, 1]]
    convex_loss = torch.mul(normalize(coord0 - coord1, 1), normal1) + torch.mul(normalize(coord1 - coord0, 1), normal0) 
    return -torch.mean(convex_loss)

def compute_move_loss(pre_coord, coord):
    move_loss = torch.mean(torch.sum((pre_coord - coord) ** 2, 1))
    return move_loss

def compute_normal_loss(normal, min_idx, edge, edge_vec):
    normal = normal.squeeze(0)
    min_idx = min_idx.squeeze(0)
    edge = edge.transpose(0, 1)
    normal_vec = normal[min_idx][edge[:, 0]]
    normalize = F.normalize
    cos_normal = torch.mul(normalize(normal_vec, dim=1),
                           normalize(edge_vec, dim=1))
    cos_normal_loss = torch.mean(torch.abs(torch.sum(cos_normal, 1)))
    return cos_normal_loss

def compute_edge_loss(pos, edge):

    assert(len(pos.shape) == 2)
    assert(pos.shape[-1] == 3)
    assert(len(edge.shape) == 2)
    assert(edge.shape[-1] == 2)

    coord0 = pos[edge[:, 0]]
    coord1 = pos[edge[:, 1]]
    edge_vec = coord0 - coord1
    return edge_vec, torch.mean(torch.sum(edge_vec ** 2, 1))

def compute_symm_loss(pos, symm_edge):
   
    assert(len(pos.shape) == 2)
    assert(pos.shape[-1] == 3)
    assert(len(symm_edge.shape) == 2)
    assert(symm_edge.shape[-1] == 2)

    coord0 = pos[symm_edge[:, 0]]
    coord0[:, 0] = -1. * coord0[:, 0]
    coord1 = pos[symm_edge[:, 1]]
    symm_vec = coord0 - coord1
    return torch.mean(torch.sum(symm_vec ** 2, 1))

def compute_chamfer_loss(gt_point, coord):
    chamfer_dist = ChamferDistance()
    return chamfer_dist(gt_point, coord)

def add_dict_loss(loss_dict, loss_name, value):
    if loss_name in loss_dict.keys():
        loss_dict[loss_name] += value
    else:
        loss_dict[loss_name] = value

def compute_image_loss(mask, gt_normal, pre_normal):   
    mse = F.mse_loss
    return mse(pre_normal*mask, gt_normal * mask)

def compute_loss(param, init_mesh, input_dict, output_dict):
    
    B = input_dict['render_img'].shape[0]
    size_list = [(input_dict['batch'] == i).sum() for i in range(B)]
    vert_list = input_dict['vertex'].split(size_list, 0)

    if input_dict['normal'] is not None:
        norm_lists = input_dict['normal'].split(size_list, 0)
    
    if input_dict['face'] is not None:
        face_mask = input_dict['batch'][input_dict['face'][0]]
        face_size_list = [(face_mask == i).sum() for i in range(B)]
        face_list = input_dict['face'].t().split(face_size_list, 0)

    total_loss = 0
    loss_dict = {}
    sample_points_list = []

    for i in range(3):
        for j in range(B):
            symm_loss = compute_symm_loss(output_dict['coords'][i][j], init_mesh.symm_edge0.t())
            edge_vec, edge_loss = compute_edge_loss(output_dict['coords'][i][j], output_dict['edges'][i][j].t())
            
            # if input_dict['normal'] is None:
            #     face_idx = face_list[j]
            #     for k in range(j):
            #         face_idx = face_idx - size_list[k]
            #     norm_list = compute_vert_normal(vert_list[j], face_idx)
            # else:
            #     norm_list = norm_lists[j]
            
            sample_points, sample_normals = reparam_sample(output_dict['coords'][i][j], output_dict['faces'][i][j], 10000) 
            if i == 2:
                sample_points_list.append(sample_points)

            if param.use_diff_sub and i == 2:
                gt_detail = input_dict['vertex'][input_dict['batch'] == j][:5000]
                gt_sample = torch.cat([input_dict['sample'][j], gt_detail] ,0).unsqueeze(0)
            else:
                gt_sample = input_dict['sample'][j].unsqueeze(0)

            chamfer_loss = compute_chamfer_loss(
                gt_sample, sample_points.unsqueeze(0))
            dist1, dist2, idx1, idx2 = chamfer_loss
            point_loss = torch.mean(dist1) + torch.mean(dist2)
            idx1 = idx1.squeeze(0)
            idx2 = idx2.squeeze(0)
            # norm_loss = - torch.mean(torch.abs(torch.sum(norm_list * sample_normals[idx1.long()], 1))) - torch.mean(torch.abs(torch.sum(sample_normals * norm_list[idx2.long()], 1)))

            # convex_loss = compute_convex_loss(output_dict['coords'][i][j], output_dict['faces'][i][j], output_dict['edges'][i][j].t())

            # laplace loss
            if i == 0:
                if param.use_new_laploss:
                    laplace_loss = compute_laplace_loss_new(output_dict['coords'][i][j], init_mesh.lap_idx[i])
                else:
                    laplace_loss = 0.2 * compute_laplace_loss(init_mesh.vertices, output_dict['coords'][i][j], init_mesh.lap_idx[i])
                move_loss = None

            else:
                if not param.use_diff_sub:
                    if param.use_new_laploss:
                        laplace_loss = compute_laplace_loss_new(output_dict['coords'][i][j], init_mesh.lap_idx[i])
                    else:
                        laplace_loss = compute_laplace_loss(output_dict['coords'][i + 2][j], output_dict['coords'][i][j], init_mesh.lap_idx[i])
                else:
                    laplace_loss = compute_laplace_loss_new_sub(output_dict['coords'][i][j], output_dict['edges'][i][j])

                move_loss = compute_move_loss(output_dict['coords'][i + 2][j], output_dict['coords'][i][j])

            total_loss += param.weight_dict['symm_loss_weight'] * symm_loss
            add_dict_loss(loss_dict, 'symm_loss{}'.format(i), param.weight_dict['symm_loss_weight'] * symm_loss.item())

            total_loss += param.weight_dict['point_loss_weight'] * point_loss
            add_dict_loss(loss_dict, 'point_loss{}'.format(i), param.weight_dict['point_loss_weight'] * point_loss.item())

            if not (param.use_diff_sub and i == 2):
                total_loss += param.weight_dict['edge_loss_weight'] * edge_loss
                add_dict_loss(loss_dict, 'edge_loss{}'.format(i), param.weight_dict['edge_loss_weight'] * edge_loss.item())

            # total_loss += param.weight_dict['norm_loss_weight'] * norm_loss
            # add_dict_loss(loss_dict, 'norm_loss{}'.format(i), param.weight_dict['norm_loss_weight'] * norm_loss.item())

            total_loss += param.weight_dict['laplace_loss_weight'] * laplace_loss
            add_dict_loss(loss_dict, 'laplace_loss{}'.format(i), param.weight_dict['laplace_loss_weight'] * laplace_loss.item())
            
            if move_loss is not None:
                total_loss += param.weight_dict['move_loss_weight'] * move_loss
                add_dict_loss(loss_dict, 'move_loss{}'.format(i), param.weight_dict['move_loss_weight'] * move_loss.item())

    return total_loss, loss_dict, vert_list, sample_points_list

def compute_fscore(input_dict, output_dict, threshold):
    
    B = input_dict['render_img'].shape[0]
    sample_num = 10000

    sample_lists = []
    for i in range(B):
        sample, _ = reparam_sample(output_dict['coords'][2][i], output_dict['faces'][2][i], sample_num)
        sample_lists.append(sample)
    
    sample_verts = torch.stack(sample_lists, dim=0)
    gt = input_dict['sample']

    dist1, dist2, _, _ = compute_chamfer_loss(sample_verts, gt)

    dist1 = dist1 * 0.57 < threshold
    dist2 = dist2 * 0.57 < threshold

    precision = 100.0 * dist2.sum(1).float() / (dist2.shape[1] + 1e-8)
    recall = 100.0 * dist1.sum(1).float() / (dist1.shape[1] + 1e-8)
    f_score = 2 * (precision * recall)/(precision + recall + 1e-8)

    return f_score.sum()


def compute_psgn_loss(input_dict, pred_points):
    B = input_dict['render_img'].shape[0]

    total_loss = 0

    for j in range(B):
        gt_points = input_dict['vertex'][input_dict['batch'] == j][:5000].unsqueeze(0)
        chamfer_loss = compute_chamfer_loss(gt_points, pred_points[j].unsqueeze(0))
        dist1, dist2, idx1, idx2 = chamfer_loss
        # point_loss = torch.mean(dist1) + torch.mean(dist2)
        # point_loss = torch.mean(dist1)
        point_loss = torch.mean(dist1) + 0.5 * torch.mean(dist2)
        total_loss += point_loss
    return total_loss