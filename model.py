import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from common import normalize_imagenet
from common import export_pointcloud
import tempfile
import subprocess
import os
import trimesh


class ConvEncoder(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=128):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_out = nn.Linear(512, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = net.view(batch_size, 512, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out


class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet34(nn.Module):
    r''' ResNet-34 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet50(nn.Module):
    r''' ResNet-50 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet101(nn.Module):
    r''' ResNet-101 encoder network.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet101(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class PSGN_Cond(nn.Module):
    r''' Point Set Generation Network encoding network.

    The PSGN conditioning network from the original publication consists of
    several 2D convolution layers. The intermediate outputs from some layers
    are used as additional input to the encoder network, similar to U-Net.

    Args:
        c_dim (int): output dimension of the latent embedding
    '''
    def __init__(self, c_dim=512):
        super().__init__()
        actvn = nn.ReLU()
        num_fm = int(c_dim/32)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, num_fm, 3, 1, 1), actvn,
            nn.Conv2d(num_fm, num_fm, 3, 1, 1), actvn)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(num_fm, num_fm*2, 3, 2, 1), actvn,
            nn.Conv2d(num_fm*2, num_fm*2, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*2, num_fm*2, 3, 1, 1), actvn)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(num_fm*2, num_fm*4, 3, 2, 1), actvn,
            nn.Conv2d(num_fm*4, num_fm*4, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*4, num_fm*4, 3, 1, 1), actvn)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(num_fm*4, num_fm*8, 3, 2, 1), actvn,
            nn.Conv2d(num_fm*8, num_fm*8, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*8, num_fm*8, 3, 1, 1), actvn)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(num_fm*8, num_fm*16, 3, 2, 1), actvn,
            nn.Conv2d(num_fm*16, num_fm*16, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*16, num_fm*16, 3, 1, 1), actvn)
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(num_fm*16, num_fm*32, 3, 2, 1), actvn,
            nn.Conv2d(num_fm*32, num_fm*32, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*32, num_fm*32, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*32, num_fm*32, 3, 1, 1), actvn)
        self.conv_block7 = nn.Sequential(
            nn.Conv2d(num_fm*32, num_fm*32, 5, 2, 2), actvn)

        self.trans_conv1 = nn.Conv2d(num_fm*8, num_fm*4, 3, 1, 1)
        self.trans_conv2 = nn.Conv2d(num_fm*16, num_fm*8, 3, 1, 1)
        self.trans_conv3 = nn.Conv2d(num_fm*32, num_fm*16, 3, 1, 1)

    def forward(self, x, return_feature_maps=True):
        r''' Performs a forward pass through the network.

        Args:
            x (tensor): input data
            return_feature_maps (bool): whether intermediate feature maps
                    should be returned
        '''
        feature_maps = []

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        feature_maps.append(self.trans_conv1(x))

        x = self.conv_block5(x)
        feature_maps.append(self.trans_conv2(x))

        x = self.conv_block6(x)
        feature_maps.append(self.trans_conv3(x))

        x = self.conv_block7(x)

        if return_feature_maps:
            return x, feature_maps
        return x


class PSGN_2Branch(nn.Module):
    r''' The 2-Branch decoder of the Point Set Generation Network.

    The latent embedding of the image is passed through a fully-connected
    branch as well as a convolution-based branch which receives additional
    input from the conditioning network.
    '''
    def __init__(self, dim=3, c_dim=512, n_points=1024):
        r''' Initialisation.

        Args:
            dim (int): dimension of the output points (e.g. 3)
            c_dim (int): dimension of the output of the conditioning network
            n_points (int): number of points to predict

        '''
        super().__init__()
        # Attributes
        actvn = nn.ReLU()
        self.actvn = actvn
        self.dim = dim
        num_fm = int(c_dim/32)
        conv_c_in = 32 * num_fm
        fc_dim_in = 3*4*conv_c_in  # input image is downsampled to 3x4
        fc_pts = n_points - 768  # conv branch has a fixed output of 768 points

        # Submodules
        self.fc_branch = nn.Sequential(nn.Linear(fc_dim_in, fc_pts*dim), actvn)
        self.deconv_1 = nn.ConvTranspose2d(c_dim, num_fm*16, 5, 2, 2, 1)
        self.deconv_2 = nn.ConvTranspose2d(num_fm*16, num_fm*8, 5, 2, 2, 1)
        self.deconv_3 = nn.ConvTranspose2d(num_fm*8, num_fm*4, 5, 2, 2, 1)
        # TODO: unused, remove? (keep it for now to load old checkpoints)
        self.deconv_4 = nn.ConvTranspose2d(num_fm*4, 3, 5, 2, 2, 1)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_fm*16, num_fm*16, 3, 1, 1), actvn)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(num_fm*8, num_fm*8, 3, 1, 1), actvn)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(num_fm*4, num_fm*4, 3, 1, 1), actvn)
        self.conv_4 = nn.Conv2d(num_fm*4, dim, 3, 1, 1)

    def forward(self, c):
        x, feature_maps = c
        batch_size = x.shape[0]

        fc_branch = self.fc_branch(x.view(batch_size, -1))
        fc_branch = fc_branch.view(batch_size, -1, 3)

        conv_branch = self.deconv_1(x)
        conv_branch = self.actvn(torch.add(conv_branch, feature_maps[-1]))

        conv_branch = self.conv_1(conv_branch)
        conv_branch = self.deconv_2(conv_branch)
        conv_branch = self.actvn(torch.add(conv_branch, feature_maps[-2]))

        conv_branch = self.conv_2(conv_branch)
        conv_branch = self.deconv_3(conv_branch)
        conv_branch = self.actvn(torch.add(conv_branch, feature_maps[-3]))

        conv_branch = self.conv_3(conv_branch)
        conv_branch = self.conv_4(conv_branch)
        conv_branch = conv_branch.view(batch_size, -1, self.dim)

        output = torch.cat([fc_branch, conv_branch], dim=1)
        return output


class Decoder(nn.Module):
    r''' Simple decoder for the Point Set Generation Network.

    The simple decoder consists of 4 fully-connected layers, resulting in an
    output of 3D coordinates for a fixed number of points.

    Args:
        dim (int): The output dimension of the points (e.g. 3)
        c_dim (int): dimension of the input vector
        n_points (int): number of output points
    '''
    def __init__(self, dim=3, c_dim=128, n_points=1024):
        super().__init__()
        # Attributes
        self.dim = dim
        self.c_dim = c_dim
        self.n_points = n_points

        # Submodules
        self.actvn = F.relu
        self.fc_0 = nn.Linear(c_dim, 512)
        self.fc_1 = nn.Linear(512, 512)
        self.fc_2 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, dim*n_points)

    def forward(self, c):
        batch_size = c.size(0)

        net = self.fc_0(c)
        net = self.fc_1(self.actvn(net))
        net = self.fc_2(self.actvn(net))
        points = self.fc_out(self.actvn(net))
        points = points.view(batch_size, self.n_points, self.dim)

        return points


class PSGN(nn.Module):
    r''' The Point Set Generation Network.

    For the PSGN, the input image is first passed to a encoder network,
    e.g. restnet-18 or the CNN proposed in the original publication. Next,
    this latent code is then used as the input for the decoder network, e.g.
    the 2-Branch model from the PSGN paper.

    Args:
        decoder (nn.Module): The decoder network
        encoder (nn.Module): The encoder network
    '''

    def __init__(self, decoder, encoder):
        super().__init__()
        self.decoder = decoder       # simple: Decoder;  # psgn_2branch: PSGN_2Branch
        self.encoder = encoder       # resnet18

    def forward(self, x):
        c = self.encoder(x)
        points = self.decoder(c)
        return points


# def create_psgn_official_sample(train=True):
#     r'''
#     official implements
#     encoder: PSGN_Cond
#     decoder: Decoder
#     :return:
#     '''
#     decoder = Decoder()
#     encoder = PSGN_Cond()
#     model = PSGN(decoder, encoder)
#     if train:
#         model = model.train()
#     else:
#         model = model.eval()
#     return model
#
#
# def create_psgn_official_2branch(train=True):
#     r'''
#     official implements
#     encoder: PSGN_Cond
#     decoder: PSGN_2Branch
#     :return:
#     '''
#     decoder = PSGN_2Branch()
#     encoder = PSGN_Cond()
#     model = PSGN(decoder, encoder)
#     if train:
#         model = model.train()
#     else:
#         model = model.eval()
#     return model


def create_model(train=True):
    r'''
    occupancy_networks setting
    encoder: Resnet18
    decoder: Decoder(simple)
    c_dim: 256
    z_dim: 0
    img_size: 224
    pointcloud_target_n: 1024
    n_x: 128
    n_z: 1
    batch_size: 64
    :return:
    '''
    decoder = Decoder(dim=3, c_dim=256, n_points=5000)
    encoder = Resnet18(c_dim=256)
    model = PSGN(decoder, encoder)
    if train:
        model = model.train()
    else:
        model = model.eval()
    return model


def create_psgn_occu(train=True):
    r'''
    occupancy_networks setting
    encoder: Resnet18
    decoder: Decoder(simple)
    c_dim: 256
    z_dim: 0
    img_size: 224
    pointcloud_target_n: 1024
    n_x: 128
    n_z: 1
    batch_size: 64
    :return:
    '''
    decoder = Decoder(dim=3, c_dim=256)
    encoder = Resnet18(c_dim=256)
    model = PSGN(decoder, encoder)
    if train:
        model = model.train()
    else:
        model = model.eval()
    return model


class Generator3D(object):
    r''' Generator Class for Point Set Generation Network.

    While for point cloud generation the output of the network if used, for
    mesh generation, we perform surface reconstruction in the form of ball
    pivoting. In practice, this is done by using a respective meshlab script.

    Args:
        model (nn.Module): Point Set Generation Network model
        device (PyTorch Device): the PyTorch devicd
    '''
    def __init__(self, model, device=None,
                 knn_normals=5, poisson_depth=10):
        self.model = model.to(device)
        self.device = device
        # TODO Can we remove these variables?
        self.knn_normals = knn_normals
        self.poisson_depth = poisson_depth

    def generate_pointcloud(self, data):
        r''' Generates a point cloud by simply using the output of the network.

        Args:
            data (tensor): input data
        '''
        self.model.eval()
        device = self.device

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        with torch.no_grad():
            points = self.model(inputs).squeeze(0)

        points = points.cpu().numpy()
        return points

    def generate_mesh(self, data):
        r''' Generates meshes by performing ball pivoting on the output of the network.

        Args:
            data (tensor): input data
        '''
        self.model.eval()
        device = self.device

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        with torch.no_grad():
            points = self.model(inputs).squeeze(0)

        points = points.cpu().numpy()
        mesh = meshlab_poisson(points)

        return mesh


FILTER_SCRIPT_RECONSTRUCTION = '''
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Surface Reconstruction: Ball Pivoting">
  <Param value="0" type="RichAbsPerc" max="1.4129" name="BallRadius" description="Pivoting Ball radius (0 autoguess)" min="0" tooltip="The radius of the ball pivoting (rolling) over the set of points. Gaps that are larger than the ball radius will not be filled; similarly the small pits that are smaller than the ball radius will be filled."/>
  <Param value="20" type="RichFloat" name="Clustering" description="Clustering radius (% of ball radius)" tooltip="To avoid the creation of too small triangles, if a vertex is found too close to a previous one, it is clustered/merged with it."/>
  <Param value="90" type="RichFloat" name="CreaseThr" description="Angle Threshold (degrees)" tooltip="If we encounter a crease angle that is too large we should stop the ball rolling"/>
  <Param value="false" type="RichBool" name="DeleteFaces" description="Delete intial set of faces" tooltip="if true all the initial faces of the mesh are deleted and the whole surface is rebuilt from scratch, other wise the current faces are used as a starting point. Useful if you run multiple times the algorithm with an incrasing ball radius."/>
 </filter>
</FilterScript>
'''


def meshlab_poisson(pointcloud):
    r''' Runs the meshlab ball pivoting algorithm.

    Args:
        pointcloud (numpy tensor): input point cloud
    '''
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'script.mlx')
        input_path = os.path.join(tmpdir, 'input.ply')
        output_path = os.path.join(tmpdir, 'out.off')

        # Write script
        with open(script_path, 'w') as f:
            f.write(FILTER_SCRIPT_RECONSTRUCTION)

        # Write pointcloud
        export_pointcloud(pointcloud, input_path, as_text=False)

        # Export
        env = os.environ
        subprocess.Popen('meshlabserver -i ' + input_path + ' -o '
                         + output_path + ' -s ' + script_path,
                         env=env, cwd=os.getcwd(), shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         ).wait()
        mesh = trimesh.load(output_path, process=False)

    return mesh



if __name__ == '__main__':

    image = torch.randn(32, 3, 224, 224)
    print("image: ", image.shape)

    model = create_psgn_occu()
    points = model(image)
    print("points: ", points.shape)

    # model = create_psgn_official_sample()
    # points = model(image)
    # print("points: ", points.shape)

    # model = create_psgn_official_2branch()
    # points = model(image)
    # print("points: ", points.shape)




