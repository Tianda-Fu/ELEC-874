import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as vutils
from torchvision import datasets, transforms
from torch.autograd import Variable


# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, d * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(d * 2, d, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            nn.ConvTranspose2d(d, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        # self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        # self.deconv1_bn = nn.BatchNorm2d(d * 8)
        # self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        # self.deconv2_bn = nn.BatchNorm2d(d * 4)
        # self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        # self.deconv3_bn = nn.BatchNorm2d(d * 2)
        # self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        # self.deconv4_bn = nn.BatchNorm2d(d)
        # self.deconv5 = nn.ConvTranspose2d(d, 64, 4, 2, 1)
        # self.deconv5_bn = nn.BatchNorm2d(64)
        # self.deconv6 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        # self.tanh = nn.Tanh()

    # # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, Input):
        # # x = F.relu(self.deconv1(input))
        # x = F.relu(self.deconv1_bn(self.deconv1(input)))
        # x = F.relu(self.deconv2_bn(self.deconv2(x)))
        # x = F.relu(self.deconv3_bn(self.deconv3(x)))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.relu(self.deconv5_bn(self.deconv5(x)))
        # x = self.tanh(self.deconv6(x))

        return self.main(Input)


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, d, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d, d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d * 2, d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d * 4, d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        # self.conv0 = nn.Conv2d(3, 64, 4, 2, 1)
        # self.conv1 = nn.Conv2d(64, d, 4, 2, 1)
        # self.conv1_bn = nn.BatchNorm2d(d)
        # self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        # self.conv2_bn = nn.BatchNorm2d(d * 2)
        # self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        # self.conv3_bn = nn.BatchNorm2d(d * 4)
        # self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        # self.conv4_bn = nn.BatchNorm2d(d * 8)
        # self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)
        # self.sigmoid = nn.Sigmoid()

    # # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, Input):
        # x = F.leaky_relu(self.conv0(input), 0.2)
        # x = F.leaky_relu(self.conv1_bn(self.conv1(x)), 0.2)
        # x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        # x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        # x = self.sigmoid(self.conv5(x))

        return self.main(Input)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def normal_init(m, mean, std):
#     if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
#         m.weight.data.normal_(mean, std)
#         m.bias.data.zero_()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)  # fixed noise
fixed_z_ = fixed_z_.cuda()


def show_result(num_epoch, show=False, save=False, path='result.png', isFix=False):
    z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
    z_ = z_.cuda()

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5 * 5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# training parameters
batch_size = 128
lr_g = 0.0001
lr_d = 0.0004
train_epoch = 20

# data_loader
img_size = 128
isCrop = False
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
data_dir = 'data/'  # this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True, num_workers=0)
temp = plt.imread(train_loader.dataset.imgs[0][0])
if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
    sys.stderr.write('Error! image size is not 128 x 128!')
    sys.exit(1)

# show the images
# real_batch = next(iter(train_loader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].cuda[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# print(train_loader.shape())

# network
G = generator(128).to(device)
D = discriminator(128).to(device)
G.apply(weights_init)
D.apply(weights_init)


# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.


# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('DCGAN_results'):
    os.mkdir('DCGAN_results')
if not os.path.isdir('DCGAN_results/Random_results'):
    os.mkdir('DCGAN_results/Random_results')
if not os.path.isdir('DCGAN_results/Fixed_results'):
    os.mkdir('DCGAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('Training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch + 1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch + 1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    num_iter = 0

    epoch_start_time = time.time()
    for i, img in enumerate(train_loader, 0):
        # train discriminator D
        D.zero_grad()

        if isCrop:
            img = img[:, :, 22:86, 22:86]

        img = img[0].to(device)
        mini_batch = img.size(0)

        label = torch.full((mini_batch,), real_label, dtype=torch.float, device=device)

        # y_real_ = torch.ones(mini_batch)
        # y_fake_ = torch.zeros(mini_batch)

        # print(y_real_.size())


        # y_real_ = y_real_.cuda()
        # y_fake_ = y_fake_.cuda()
        D_result = D(img).view(-1)
        D_real_loss = BCE_loss(D_result, label)
        D_real_loss.backward()
        D_real_score = D_result.mean().item()

        z_ = torch.randn(mini_batch, 100, 1, 1, device=device)
        # z_ = z_.cuda()
        G_result = G(z_)

        label.fill_(fake_label)

        D_result = D(G_result.detach()).view(-1)
        D_fake_loss = BCE_loss(D_result, label)
        D_fake_loss.backward()
        D_fake_score = D_result.mean().item()

        D_train_loss = D_real_loss + D_fake_loss

        # D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data.item())

        # train generator G : maximize log(D(G(z)))
        G.zero_grad()

        label.fill_(real_label)

        # z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        # z_ = z_.cuda()

        # G_result = G(z_)
        D_result = D(G_result).view(-1)
        G_train_loss = BCE_loss(D_result, label)
        G_train_loss.backward()
        G_train_score = G_train_loss.mean().item()
        G_optimizer.step()

        G_losses.append(G_train_loss.data.item())

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
    (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
    torch.mean(torch.FloatTensor(G_losses))))
    p = 'DCGAN_results/Random_results/DCGAN_' + str(epoch + 1) + '.png'
    fixed_p = 'DCGAN_results/Fixed_results/DCGAN_' + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=p, isFix=False)
    show_result((epoch + 1), save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "DCGAN_results/generator_param.pkl")
torch.save(D.state_dict(), "DCGAN_results/discriminator_param.pkl")
with open('DCGAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='DCGAN_results/DCGAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'DCGAN_results/Fixed_results/DCGAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('DCGAN_results/generation_animation.gif', images, fps=5)