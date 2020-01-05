import time
import torch
from tqdm import tqdm
from option import create_parser
from dataset import creare_dataset
from logger import create_logger
from model import create_model
from saver import create_saver
from torch_geometric.data import DataLoader, DataListLoader, Data
import torch_geometric

from loss_psgn import chamfer_distance
from loss.loss_function import compute_psgn_loss
import visualize as vis
import os


def make_input_dict(param, data):
    input_dict = {}
    if param.data_type == 'p2m':
        input_dict['render_img'] = data.y
        input_dict['normal_img'] = None
        input_dict['mask'] = None
        input_dict['proj_mat'] = None
        input_dict['batch'] = data.batch
        input_dict['vertex'] = data.pos
        input_dict['face'] = None
        input_dict['normal'] = data.norm
        input_dict['sample'] = None

    elif param.data_type == 'shapenet':
        input_dict['render_img'] = data.render_img
        input_dict['normal_img'] = None
        input_dict['mask'] = None
        input_dict['proj_mat'] = data.proj_mat
        input_dict['batch'] = data.batch
        input_dict['vertex'] = data.pos   # simplify_data pos
        input_dict['face'] = data.face
        input_dict['normal'] = None
        input_dict['sample'] = data.sample_pos

    return input_dict


def run_eval(test_loader, model, logger, n_epoch, device, param):
    n_iter = 0

    print('start evaluating for epoch ', n_epoch)
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)

            input_dict = make_input_dict(param, data)
            # print("eval: input_dict['render_img']: ", input_dict['render_img'].shape)
            pred_points = model(input_dict['render_img'])
            # print("eval: pred_points: ", pred_points.shape)
            gt_points = input_dict['vertex']
            # print("eval: gt_points: ", gt_points.shape)
            gt_sample = input_dict['sample']
            # print("eval: gt_sample: ", gt_sample.shape)

            # total_loss = chamfer_distance(gt_points, pred_points)
            total_loss = compute_psgn_loss(input_dict, pred_points).mean()

            n_iter += 1

            if logger is not None:
                if n_iter % param.eval_mesh_freq_iter == 0:
                    logger.add_loss('eval total loss', total_loss.item(), n_iter)
                    logger.add_image('eval input image', input_dict['render_img'], n_iter)
                    logger.save_mesh('eval_gt_pcl', input_dict['vertex'][input_dict['batch'] == 0], n_iter)
                    logger.save_mesh('eval_pre_pcl3', pred_points[0], n_iter)

                # if param.vis:
                #     inputs = input_dict['render_img'].data.cpu()
                #     pred_points = pred_points.data.cpu().numpy()
                #     gt_points = gt_points.data.cpu().numpy()
                #     batch_size = inputs.size(0)
                #     for i in range(batch_size):
                #         input_img_path = os.path.join("vis", '%03d_in.png' % i)
                #         vis.visualize_data(
                #             inputs[i], 'img', input_img_path)
                #         out_file = os.path.join("vis", '%03d.png' % i)
                #         out_file_gt = os.path.join("vis", '%03d_gt.png' % i)
                #         vis.visualize_pointcloud(pred_points[i], out_file=out_file)
                #         vis.visualize_pointcloud(gt_points[input_dict['batch'].data.cpu().numpy() == i], out_file=out_file_gt)
                #     pass
    print('finish evaluating for epoch ', n_epoch)
    # if logger is not None:
    #     logger.add_eval('f score', f_score / test_loader.dataset.__len__(), n_epoch)
    # print('{} epoch f1 score = {}'.format(n_epoch, f_score / test_loader.dataset.__len__()))


def train(param):
    # use cuda
    device = torch.device(
        'cuda:{}'.format(param.gpu_ids[0]) if param.use_cuda and torch.cuda.is_available() else 'cpu')
    # dataset
    train_dataset = creare_dataset(
        param.data_type, param.data_root, categories=param.train_category, train=True)
    test_dataset = creare_dataset(
        param.data_type, param.data_root, categories=param.eval_category, train=False)
    # dataloader
    train_loader = DataLoader(train_dataset.dataset, batch_size=param.batch_size, shuffle=True,
                              num_workers=param.num_workers, pin_memory=param.pin_mem)
    test_loader = DataLoader(test_dataset.dataset, batch_size=param.batch_size, shuffle=False,
                             num_workers=param.num_workers, pin_memory=param.pin_mem)
    # logger
    logger = None
    if param.use_logger:
        logger = create_logger(param.log_path + '_' + param.name)

    model = create_model(train=True)
    # checkpoint
    saver = create_saver(param.save_path + '_' + param.name)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)
    # iteration
    n_iter = 0
    # parallel setting
    if param.muti_gpu:
        model = torch.nn.DataParallel(model, param.gpu_ids)
    model = model.to(device)

    for epoch in range(param.epoch):
        print('start traning for epoch ', epoch)
        for data in tqdm(train_loader):

            # read data
            data = data.to(device)
            optimizer.zero_grad()

            # collect input
            input_dict = make_input_dict(param, data)
            # print("train: input_dict['render_img']: ", input_dict['render_img'].shape)
            pred_points = model(input_dict['render_img'])
            # print("train: pred_points: ", pred_points.shape)
            # gt_points = input_dict['vertex']
            # print("train:  gt_points: ", gt_points.shape)
            gt_sample = input_dict['sample']
            # print("train: gt_sample: ", gt_sample.shape)

            # total_loss = chamfer_distance(gt_points, pred_points)
            total_loss = compute_psgn_loss(input_dict, pred_points).mean()

            total_loss.backward()
            optimizer.step()

            n_iter += 1

            if logger is not None:
                if n_iter % param.loss_freq_iter == 0:
                    logger.add_loss('train total loss', total_loss.item(), n_iter)

                if n_iter % param.check_freq_iter == 0:
                    logger.add_gradcheck(model.named_parameters(), n_iter)
                    logger.add_image('input image', input_dict['render_img'], n_iter)

                if n_iter % param.train_mesh_freq_iter == 0:
                    logger.save_mesh('train_gt_pcl', input_dict['vertex'][input_dict['batch'] == 0], n_iter)
                    logger.save_mesh('train_pre_pcl3', pred_points[0], n_iter)


        print('finish traning for epoch ', epoch)
        if isinstance(model, torch.nn.DataParallel):
            saver.save_model(model.module.state_dict(), epoch)
        else:
            saver.save_model(model.state_dict(), epoch)
        # saver.save_optimizer(optimizer.state_dict(), epoch)

        print('finish saving checkpoint for epoch ', epoch)
        if n_iter % param.eval_freq_epoch == 0:
            run_eval(
                test_loader, model, logger, epoch, device, param)



if __name__ == '__main__':
    # parameters
    param = create_parser()
    # start training
    train(param)
