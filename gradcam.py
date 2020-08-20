import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from efficientnet_pytorch import EfficientNet
from torch import nn
import os
import shutil
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0")

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            #根据你想测试的网络自行添加网络模块
            # elif name == "_blocks":
            #     print("in ************&&&&")
            #     for lmodule in module:
            #        print("%%%%%")
            #        print(lmodule)
            #        x = lmodule(x) 
            else:
                if "_fc" in name.lower():
                    np.squeeze(x) 
                    x = np.squeeze(x)
                    x = module(x)

                else:
                    x = module(x)
        
        return target_activations, x


def preprocess_image(img):

    # means = [0.485, 0.456, 0.406]
    # stds = [0.229, 0.224, 0.225]
    print(img.shape,"in preprocess")
    means = [np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])]
    stds = [np.std(img[:,:,0]), np.std(img[:,:,1]), np.std(img[:,:,2])]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    # print("input shape", input.shape, "&&&&&&&&&&&&", img.shape)
    return input


def show_cam_on_image(img, mask, save_path, filename):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cv2.imwrite(f"{save_path}/{filename}_cam1.jpg", img * 255)
    cv2.imwrite(f"{save_path}/{filename}_cam2.jpg", heatmap * 255)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(f"{save_path}/{filename}_cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='/home/xhzhu/data/my_siim/512x512-dataset-melanoma/512x512-dataset-melanoma/ISIC_0000000.jpg',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

def load(path, net_num, gpus=False):
    checkpoint_path = path
    checkpoint = torch.load(checkpoint_path)
    if gpus:
        net = torch.nn.DataParallel(get_net(net_num)).cuda()
    else:
        net = get_net(net_num).cuda()
    net.load_state_dict(checkpoint)
    net.eval()
    return net

def get_net(f=3):
    #根据需求导入网络模型
    if f == 3:
        #efficientb3
        print("3333333333333333333")
        net = EfficientNet.from_pretrained('efficientnet-b3')
        net._fc = nn.Linear(in_features=1536, out_features=2, bias=True)
        return net
    elif f == 5:
        #efficientb5
        print("5555555")
        net = EfficientNet.from_pretrained('efficientnet-b5')
        net._fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        return net
    else:
        #efficientb6
        print("666666")
        net = EfficientNet.from_pretrained('efficientnet-b6')
        net._fc = nn.Linear(in_features=2304, out_features=2, bias=True)
        return net

# net = get_net().cuda()
# net = load("/home/xhzhu/mycode/siim_code/fold3/best-score-checkpoint-018epoch.bin")
# image path:/home/xhzhu/data/my_siim/512x512-dataset-melanoma/512x512-dataset-melanoma/ISIC_0000000.jpg

"""
两个函数分别针对普通图片目录和用pytorch制作好的dataloader来才获取cam热图
"""
def dir_test(model_path, test_path, save_path, use_cuda=True, gpus=False, net_num=3):
    #input image dir
    model = load(model_path, net_num, gpus)
    if gpus:
        model = model.module
    #print(model)
    #efficientnet
    #这里需要知道你想知道网络哪一层的cam热图，一般后面接全局平均池化层
    if net_num == 3:
        grad_cam = GradCam(model=model, feature_module=model._blocks, \
                           target_layer_names=["25"], use_cuda=use_cuda)
    elif net_num ==5 :
        grad_cam = GradCam(model=model, feature_module=model._blocks, \
                           target_layer_names=["38"], use_cuda=use_cuda)
    else:
        grad_cam = GradCam(model=model, feature_module=model._blocks, \
                           target_layer_names=["44"], use_cuda=use_cuda) 
    """
    there is use origin image dir
    """
    
    num = 0
    for filename in os.listdir(test_path):
        if num > 100:
            break
        num = num + 1
        img = cv2.imread(test_path + "/" + filename)
        ig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array([ig, ig, ig])
        img = img.transpose(1,2,0)
        cv2.imwrite(f'{save_path}/{filename}.jpg', img)
        img = np.float32(cv2.resize(img, (384, 384))) / 255
        print("*****************in dir", img.shape)
        # img = np.float32(img) / 255
        input = preprocess_image(img)
        target_index = None
        mask = grad_cam(input, target_index)
        show_cam_on_image(img, mask, save_path, filename)
        #这是另外一种查看网络是否学到输入特征的方法：导向反向传播，另外还有一种反卷积也可以查看
        #这里不需要导向反向传播，所以注释了
        # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
        # #print(model._modules.items())
        # gb = gb_model(input, index=target_index)
        # gb = gb.transpose((1, 2, 0))
        # cam_mask = cv2.merge([mask, mask, mask])
        # cam_gb = deprocess_image(cam_mask*gb)
        # gb = deprocess_image(gb)
        
        # print(f'{save_path}/{filename}_gb.jpg')
        # cv2.imwrite(f'{save_path}/{filename}_gb.jpg', gb)
        # cv2.imwrite(f'{save_path}/{filename}_cam_gb.jpg', cam_gb)

def dataloader_test(model_path, dataloader, save_path, use_cuda=True, gpus=False, net_num=3):
    # input image dataloader
    model = load(model_path, net_num, gpus)
    if gpus:
        model = model.module
    #efficientnet
    if net_num == 3:
        grad_cam = GradCam(model=model, feature_module=model._blocks, \
                           target_layer_names=["25"], use_cuda=use_cuda)
    elif net_num ==5 :
        grad_cam = GradCam(model=model, feature_module=model._blocks, \
                           target_layer_names=["38"], use_cuda=use_cuda)
    else:
        grad_cam = GradCam(model=model, feature_module=model._blocks, \
                           target_layer_names=["44"], use_cuda=use_cuda) 
    """
    there is use origin image dir
    """
    num = 0 #the num is random seleced for test
    for img, target, image_name in tqdm(dataloader, total=len(dataloader)):
        if num > 1000:
            break
        num =  num + 1
        # you need focus dimension of data,the following data's batch is 1,when > 1,you need change your code
        img = np.squeeze(np.array(img))
        #print(image_names, "image name")
        #print("***** first", img.shape)
        img = np.transpose(img, (1, 2, 0))
        cv2.imwrite(f'{save_path}/{image_name[0]}.jpg', img*255)
        #print("*****", img.shape)
        input = preprocess_image(img)
        #input = img
        target_index = None
        mask = grad_cam(input, target_index)
        show_cam_on_image(img, mask, save_path, image_name[0])

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
        #print(model._modules.items())
        gb = gb_model(input, index=target_index)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask*gb)
        gb = deprocess_image(gb)
        
        print(f'{save_path}/{image_name[0]}_gb.jpg')
        cv2.imwrite(f'{save_path}/{image_name[0]}_gb.jpg', gb)
        cv2.imwrite(f'{save_path}/{image_name[0]}_cam_gb.jpg', cam_gb)
        
if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    model_path = "/home/xhzhu/mycode/siim_code/model/b3/384gray/fold0/best-score-checkpoint-013epoch.bin"
    test_path = "/home/xhzhu/data/my_siim/512x512-dataset-melanoma/512x512-dataset-melanoma"
    save_path = "/home/xhzhu/mycode/siim_code/cam_new_test"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)
    #save_path = "/home/xhzhu/mycode/siim_code/cam_dataloader_test"
    dir_test(model_path, test_path, save_path, gpus=False, net_num=3)


    #单张图片提取cam测试，需要注意的是模型必须是训练好的模型，grad cam一般是训练好了用，
    #cam才是需要改变网络结构并且在训练中提取，这里是grad cam

    # args = get_args()
    # # Can work with any model, but it assumes that the model has a
    # # feature method, and a classifier method,
    # # as in the VGG models in torchvision.
    # # model = models.vgg19(pretrained=True)
    # # f = open("vgg19.txt", "w")    # 打开文件以便写入
    # # print("Correlation is ", model,file=f)
    # # f.close
    # model = models.resnet50(pretrained=True)
    # # f = open("resnet50.txt", "w")    # 打开文件以便写入
    # # print("Correlation is ", model,file=f)
    # # f.close
    # model = net
    # # #print(net)
    # # f = open("efficient3.txt", "w")    # 打开文件以便写入
    # # print("Correlation is ", net,file=f)
    # # f.close
    # print("####################################")
    # # grad_cam = GradCam(model=model, feature_module=model.layer4, \
    # #                    target_layer_names=["2"], use_cuda=args.use_cuda)
    # grad_cam = GradCam(model=model, feature_module=model._blocks, \
    #                    target_layer_names=["25"], use_cuda=args.use_cuda)
    # # grad_cam = GradCam(model = models.vgg19(pretrained=True),feature_module=model.features,\
    # #                 target_layer_names = ["35"], use_cuda=True)
    # img = cv2.imread(args.image_path, 1)
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    # input = preprocess_image(img)

    # # If None, returns the map for the highest scoring category.
    # # Otherwise, targets the requested index.
    # target_index = None
    # mask = grad_cam(input, target_index)

    # show_cam_on_image(img, mask)

    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # print(model._modules.items())
    # gb = gb_model(input, index=target_index)
    # gb = gb.transpose((1, 2, 0))
    # cam_mask = cv2.merge([mask, mask, mask])
    # cam_gb = deprocess_image(cam_mask*gb)
    # gb = deprocess_image(gb)

    # cv2.imwrite('tts/gb.jpg', gb)
    # cv2.imwrite('tts/cam_gb.jpg', cam_gb)