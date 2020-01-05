from mesh_utils import *
import torch
import pickle
from torch_geometric.data import Dataset, Data
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import shutil
import csv
import sys
import random

print(sys.path)
sys.path.append('.')


class ShapeNetDataSet(Dataset):
    category_ids = {
        'bench': '02828884',  #
        'chair': '03001627',  #
        'lamp': '03636649',  #
        'speaker': '03691459',  #
        'firearm': '04090263',  #
        'table': '04379243',  #
        'watercraft': '04530566',  #
        'plane': '02691156',  #
        'cabinet': '02933112',  #
        'car': '02958343',  #
        'monitor': '03211117',  #
        'couch': '04256520',  #
        'cellphone': '04401088'  #
    }

    def __init__(self, root, categories=None, train=True, transform=None, pre_transform=None, pre_filter=None):
        if categories is None:
            categories = list(self.category_ids)
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories

        self.cat_dict = {}
        self.all_files = []
        self.lists = ['p2m.csv']

        # self.list_path = os.path.join('./dataset/shapenet', self.lists[0])
        self.list_path = 'p2m.csv'
        if train:
            self.flag = ['train', 'val']
        else:
            self.flag = ['test']

        for idx, cat in enumerate(self.categories):
            self.cat_dict[cat] = []
            cat_id = self.category_ids[cat]
            with open(self.list_path, 'r') as f:
                lines = csv.reader(f)
                headings = next(f)
                print('reading ', cat, ' in ', self.list_path)
                for line in tqdm(lines):
                    id = line[1].strip()
                    mesh_path = os.path.join(
                        root, id, line[-2].strip(), 'models', 'model_normalized.obj')

                    if line[-1].strip() in self.flag and id == cat_id:
                        # if os.path.exists(mesh_path):
                        self.cat_dict[cat].append(mesh_path)
                        # else:
                        #     print('data ', mesh_path, ' not exist !!!')

        # self.mesh_watertight()
        # self.mesh_simplify()
        # self.mesh_sample()
        # self.mesh_render()

        for cat, file_list in self.cat_dict.items():
            print('collecting ', cat, ' in ', self.list_path)
            count = 0
            for mesh_path in tqdm(file_list):
                splits = mesh_path.split('/')
                for n in range(4):
                    # random choose number
                    # n = random.randint(0, 16)
                    pt_file = '{}_{}_{}_{}.pt'.format(
                        cat, splits[-3], splits[-1].split('.')[-2], n)

                    # awkward way to judge the file existence
                    if os.path.exists(os.path.join(root, 'processed', pt_file)):
                        self.all_files.append(pt_file)

        print(len(self.all_files))
        super(ShapeNetDataSet, self).__init__(
            root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.lists

    @property
    def processed_file_names(self):
        return self.all_files

    def __len__(self):
        return len(self.processed_file_names)

    def mesh_watertight(self, resolution=5000):
        water_tight = os.path.join(
            './dataset/shapenet/manifold/build', 'manifold')
        for cat, file_list in self.cat_dict.items():
            print('watertight ', cat, ' in ', self.list_path)
            for mesh_path in tqdm(file_list):
                input = mesh_path
                output = mesh_path[:-4] + '_watertight.obj'
                if not os.path.exists(output) and os.path.exists(input):
                    os.system('{} {} {} {}'.format(
                        water_tight, input, output, resolution))

    def mesh_simplify(self, vert_num=5000, face_num=1000):
        simplify = os.path.join(
            './dataset/shapenet/manifold/build', 'simplify')
        for cat, file_list in self.cat_dict.items():
            print('simplify ', cat, ' in ', self.list_path)
            for mesh_path in tqdm(file_list):
                input = mesh_path[:-4] + '_watertight.obj'
                output = mesh_path[:-4] + '_simplify_5000.obj'
                if not os.path.exists(output) and os.path.exists(input):
                    os.system('{} -i {} -o {} -m -v {} -f {}'.format(
                        simplify, input, output, vert_num, face_num))

    def mesh_sample(self, vert_num=10000):
        for cat, file_list in self.cat_dict.items():
            print('sampling ', cat, ' in ', self.list_path)
            for mesh_path in tqdm(file_list):
                input = mesh_path[:-4] + '_simplify_5000.obj'
                output = mesh_path[:-4] + '_sample_10000.obj'
                if not os.path.exists(output) and os.path.exists(input):
                    simplify_mesh = load_obj(input)
                    sample_points, _ = reparam_sample(simplify_mesh.pos, simplify_mesh.face.t(), max_num=vert_num)
                    mesh = np.hstack([np.full([sample_points.shape[0], 1], 'v'), sample_points.detach().cpu().numpy()])
                    np.savetxt(output, mesh, fmt='%s', delimiter=' ')

    def mesh_render(self, render_num=16):
        for cat, file_list in self.cat_dict.items():
            print('render ', cat, ' in ', self.list_path)
            for mesh_path in tqdm(file_list):
                input = mesh_path
                paths = mesh_path.split('_')
                processdir = os.path.join(*paths[:-1]) + '_rendering/'
                outputdir = os.path.join(*paths[:-1]) + '_render_with_normal/'
                if not os.path.exists(outputdir):
                    if os.path.exists(processdir):
                        shutil.rmtree(processdir)
                    os.makedirs(processdir)
                    if os.path.exists(input):
                        os.system(
                            'blender ./dataset/shapenet/blank.blend -b -P ./dataset/shapenet/blender_render.py -- {} {} {}'.format(
                                render_num, input, processdir))
                        shutil.move(processdir, outputdir)

    def process(self):
        img_transformations = transforms.Compose(
            [transforms.Resize([224, 224]), transforms.ToTensor()])

        tmp_cnt = 0
        for cat, file_list in self.cat_dict.items():
            print('packing ', cat)
            for mesh_path in tqdm(file_list):

                splits = mesh_path.split('/')
                paths = mesh_path.split('_')

                sample_path = os.path.join(*paths[:-1]) + '_normalized_sample_10000.obj'
                if not os.path.exists(sample_path):
                    continue

                img_base_path = os.path.join(
                    *paths[:-1]) + '_render_with_normal/'
                print(img_base_path)
                for n in range(1):

                    pt_file = os.path.join(self.processed_dir, '{}_{}_{}_{}.pt'.format(
                        cat, splits[-3], splits[-1].split('.')[-2], n))
                    if os.path.exists(pt_file):
                        continue

                    img_path = os.path.join(img_base_path, str(n) + '.png')
                    if not os.path.exists(img_path):
                        shutil.rmtree(img_base_path)
                        break

                    img_array = np.array(Image.open(img_path))
                    img_array[np.where(img_array[:, :, 3] == 0)] = 255
                    img_data = img_transformations(
                        Image.fromarray(img_array[:, :, :3])).unsqueeze(0)

                    norm_path = os.path.join(
                        img_base_path, str(n) + '_normal_0000.png')
                    if not os.path.exists(norm_path):
                        shutil.rmtree(img_base_path)
                        break

                    norm_array = np.array(Image.open(norm_path))
                    norm_data = img_transformations(
                        Image.fromarray(norm_array[:, :, :3])).unsqueeze(0)

                    mask_array = np.zeros_like(norm_array[:, :, 3])
                    mask_array[np.where(norm_array[:, :, 3] != 0)] = 255
                    mask_data = img_transformations(
                        Image.fromarray(mask_array)).unsqueeze(0)

                    proj_mat_path = os.path.join(
                        img_base_path, str(n) + '.npy')
                    if not os.path.exists(proj_mat_path):
                        shutil.rmtree(img_base_path)
                        break

                    proj_mat_data = torch.Tensor(
                        np.load(proj_mat_path)).unsqueeze(0)

                    simplify_path = mesh_path[:-4] + '_simplify_5000.obj'
                    sample_path = mesh_path[:-4] + '_sample_10000.obj'

                    try:
                        simplify_data = load_obj(simplify_path)
                    except ValueError:
                        os.remove(simplify_path)
                        print('delete ', simplify_path, ' - ')
                        os.remove(sample_path)
                        print('delete ', sample_path)
                        break

                    try:
                        sample_data = load_obj(sample_path)
                    except ValueError:
                        os.remove(sample_path)
                        print('delete ', sample_path)
                        break
                    # mesh_data = load_obj(mesh_path)

                    # check normal
                    # face = mesh_data.face.transpose(0, 1)
                    # vert = [mesh_data.pos[face[:, i]] for i in range(3)]
                    # vert_id = [face[:, i] for i in range(3)]
                    # face_normal = torch.cross(vert[1] - vert[0], vert[2] - vert[0])

                    # import torch.nn.functional as F
                    # normalize = F.normalize
                    # unit_normal = normalize(face_normal)
                    # vert_normal = torch.zeros_like(mesh_data.pos)
                    # for v in vert_id:
                    #     vert_normal[v] = vert_normal[v] + unit_normal
                    # vert_normal = normalize(vert_normal)

                    # ones = torch.ones(mesh_data.pos.shape[0]).unsqueeze(1)
                    # verts = torch.cat([mesh_data.pos, ones], 1)
                    # coords = torch.mm(proj_mat_data[0], verts.transpose(0,1)).transpose(0, 1)
                    # ys = torch.clamp((coords[:, 0]/coords[:, 3] + 1) / 2 * 224, 0, 223)
                    # xs = torch.clamp(
                    #     (1 - (coords[:, 1]/coords[:, 3] + 1) / 2) * 224, 0, 223)

                    # proj_norm = torch.zeros_like(norm_data)
                    # proj_norm[:, :, xs.long(), ys.long()] = 1
                    # norm_coord = norm_data[0, :, xs.long(), ys.long()].transpose(0,1)
                    # print(mesh_path)
                    # print(norm_coord[:10])
                    # print(vert_normal[:10])

                    # from tensorboardX import SummaryWriter
                    # writer = SummaryWriter('./tmp')
                    # writer.add_image('render', img_data[0], tmp_cnt)
                    # writer.add_image('normal', norm_data[0], tmp_cnt)
                    # writer.add_image('proj', proj_norm[0], tmp_cnt)
                    # writer.close()
                    # tmp_cnt += 1

                    # if simplify_data.pos.shape[0] != 5000:
                    #     os.remove(simplify_path)
                    #     print('delete ', simplify_path, ' - ', simplify_data.pos.shape[0])
                    #     break

                    if sample_data.pos.shape[0] != 10000:
                        os.remove(sample_path)
                        print('delete ', sample_path, ' - ', sample_data.pos.shape[0])
                        break

                    data = Data(
                        render_img=img_data, proj_mat=proj_mat_data, pos=simplify_data.pos, face=simplify_data.face,
                        sample_pos=sample_data.pos.unsqueeze(0))
                    torch.save(data, pt_file)

    def get(self, idx):
        return torch.load(self.processed_paths[idx])

    def download(self):
        shutil.copy('p2m.csv', self.raw_dir)


class Dataset:
    def __init__(self, data_name, data_root, categories, train):
        if data_name.strip() == 'p2m':
            self.dataset = P2MDataSet(data_root, categories=categories, train=train)
        elif data_name.strip() == 'shapenet':
            self.dataset = ShapeNetDataSet(
                data_root, categories=categories, train=train)
        else:
            print("only support ['p2m', 'shapenet] for now")

def creare_dataset(dataset_name, data_root, categories, train):
    return Dataset(dataset_name, data_root, categories, train)


if __name__ == '__main__':
    dataset = ShapeNetDataSet('/home/lihai/Downloads/ShapeNetCore.v2')
