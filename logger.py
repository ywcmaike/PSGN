from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import torch
import shutil
import numpy as np

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.logger = SummaryWriter(log_path)
        self.once = True

    def add_loss(self, loss_name, loss_value, n_iter):
        self.logger.add_scalar(loss_name, loss_value, n_iter)

    def add_losses(self, loss_name, loss_dict, n_iter):
        self.logger.add_scalars(loss_name, loss_dict, n_iter)

    def add_image(self, img_name, img_batch, n_iter):
        img = vutils.make_grid(img_batch, normalize=True, scale_each=True)
        self.logger.add_image(img_name, img, n_iter)
    
    def save_mesh(self, mesh_name, vert, n_iter, face=None):
        mesh = np.hstack([np.full([vert.shape[0], 1], 'v'), vert.detach().cpu().numpy()])
        if face is not None:
            face = np.hstack([np.full([face.shape[0], 1], 'f'), face.cpu().numpy() + 1])
            mesh = np.vstack([mesh, face])
        out_path = os.path.join(self.log_path, '{}_{}.obj'.format(n_iter, mesh_name))
        np.savetxt(out_path, mesh, fmt='%s', delimiter=' ')

    def add_eval(self, score_name, score_value, n_epoch):
        self.logger.add_scalar(score_name, score_value, n_epoch)

    def add_graph_once(self, model, input):
        if self.once:
            self.logger.add_graph(model, input_to_model=input)
            self.once = False

    def add_figure(self, figure_name, figure, n_iter):
        self.logger.add_figure(figure_name, figure, n_iter)

    def add_gradcheck(self, named_parameters, n_iter):
        fig = plt.figure(figsize=(10, 8), dpi=100)
        plt.clf()
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                if not hasattr(p.grad, 'abs'):
                    continue
                layers.append(n[:-7])
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.plot(max_grads, alpha=0.3, color="c")
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("max / average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        self.add_figure('grad check', fig, n_iter)

    def add_projection(self, projection_name, vertices, n_iter):
        fx, fy, cx, cy = [250, 250, 112, 112]
        w0, h0 = [224, 224]
            
        bacth_size = vertices.shape[0]
        canvases = []
            
        for i in range(bacth_size):
            X = vertices[i, :, 0]
            Y = vertices[i, :, 1]
            Z = vertices[i, :, 2]
        
            h = fy * torch.div(Y, Z) + cy
            w = fx * torch.div(X, -Z) + cx

            h = torch.floor(h).long()
            w = torch.floor(w).long()

            x = torch.clamp(h, 0, 223)
            y = torch.clamp(w, 0, 223)

            canvas = torch.zeros(1, 224, 224)
            canvas[:, x, y] = 1
            canvases.append(canvas)

        project_image = torch.stack(canvases, 0)
        self.add_image(projection_name, project_image, n_iter)

    def add_projectionU(self, projection_name, vertices, proj_mat, n_iter):            
            
        bacth_size = vertices.shape[0]
        canvases = []

        for i in range(bacth_size):
            ones = torch.ones(vertices.shape[1], 1).to(vertices.device)
            point4d = torch.cat([vertices[i], ones], -1)
                
            # normalized coord [-1., 1.]
            coords = torch.mm(proj_mat[i], point4d.t()).t()

            # trnasfer to image coord [0, 1]
            x = (coords[:, 0]/coords[:, 3] + 1) / 2
            y = 1 - (coords[:, 1]/coords[:, 3] + 1) / 2

            x = torch.clamp(x * 224, 0, 223).long()
            y = torch.clamp(y * 224, 0, 223).long()

            canvas = torch.zeros(1, 224, 224)
            canvas[:, y, x] = 1
            canvases.append(canvas)

        project_image = torch.stack(canvases, 0)
        self.add_image(projection_name, project_image, n_iter)

def create_logger(log_path):
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    return Logger(log_path)
