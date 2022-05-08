from DeepImageSearch.utils.allutils import LoadImagesNormal,util
from torchvision import models
import os
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import argparse
from torch import nn
import torchvision.transforms as  T
from numpy.linalg import norm
from PIL import  Image
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing import image

class _PytorchParameters:
    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting=True):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    @staticmethod
    def show_params_learn(model, feature_extract=True):
        params_to_update = model.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)
        return params_to_update



class NoramlfeatureextractorPytroch:
    def __init__(self,config,params):
        config = util.read_yaml(config)
        artifact_dir = config['artifacts']['artifactdir']
        image_dir = config['artifacts']['image_dir']
        preprocessed = config['artifacts']['preprocessed']
        meta_data_dir = config['artifacts']['meta_data_dir']
        pytrorchfename = config['artifacts']["pytrochnomralfe"]
        pytrochnormalimgname = config['artifacts']["pytrochnomralimage"]
        self.pytorchnormalfepath = os.path.join(artifact_dir,meta_data_dir,pytrorchfename)
        self.pytrochnormalimagepath = os.path.join(artifact_dir,meta_data_dir,pytrochnormalimgname)
        self.images  = os.path.join(artifact_dir,image_dir,preprocessed)
        self.model = models.resnet18(pretrained=True)
        _PytorchParameters.set_parameter_requires_grad(self.model)
        self.model.fc = nn.Flatten()

    @classmethod
    def Transform(cls,pillowimg):
        transform = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # resent useing this preprocessed in pytroch 
            ])
        img = transform(pillowimg)
        return img

    def extract(self,img):
        img = Image.open(img).convert("RGB") # some of image not rgb so i fact this issue i found easy to overcome this is here conver image
        img = self.Transform(img)
        if len(img.shape) == 3:
            img = img[None]

        feature = self.model(img).numpy()[0] # please check shape of feature or [0] when debuggning
        return feature/norm(feature)

    def get_features_normal(self):
        dataset = LoadImagesNormal(self.images,custom='yes')
        features = []
        filenames = []

        for path in tqdm(dataset,ncols = 100,colour='MAGENTA'):
            try:
                feat = self.extract(path)
                features.append(feat)
                filenames.append(path)
            except:
                features.append(None)
                filenames.append(path)
                continue

        util.dump_pickle(self.pytorchnormalfepath,features)
        util.dump_pickle(self.pytrochnormalimagepath,filenames)

class NoramlfeatureextractorTensorflow:
    def __init__(self,config):

        config = util.read_yaml(config)
        artifact_dir = config['artifacts']['artifactdir']
        image_dir = config['artifacts']['image_dir']
        preprocessed = config['artifacts']['preprocessed']
        meta_data_dir = config['artifacts']['meta_data_dir']
        tensorflownormafename = config['artifacts']["tensoflownormalfe"]
        tensorflownormalimgname = config['artifacts']["tensorflonormalimage"]
        self.tensnormalfepath = os.path.join(artifact_dir, meta_data_dir, tensorflownormafename)
        self.tensnormalimagepath = os.path.join(artifact_dir, meta_data_dir, tensorflownormalimgname)
        self.images = os.path.join(artifact_dir, image_dir,preprocessed)
        self.model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
        self.model.trainable = False #
        self.model = tensorflow.keras.Sequential([
                                    self.model,
                                    GlobalMaxPooling2D()
                                ])
    
    def extract(self,img):
            img = image.load_img(img,target_size=(224,224))
            img = image.img_to_array(img)
            if len(img.shape) == 3:
                img = img[None]
            
            img = preprocess_input(img)
            result = self.model.predict(img).flatten()
            return result / norm(result)

    def get_features_normal(self):
        dataset = LoadImagesNormal(self.images, custom='yes')
        features = []
        filenames = []

        for path in tqdm(dataset, ncols=100, colour='YELLOW'):
            try:
                feat = self.extract(path)
                features.append(feat)
                filenames.append(path)
            except:
                features.append(None)
                filenames.append(path)
                continue

        util.dump_pickle(self.tensnormalfepath, features)
        util.dump_pickle(self.tensnormalimagepath, filenames)



class SearchImageNormalPytoch:
    def __init__(self,config,params):
        self.FE = NoramlfeatureextractorPytroch(config,params)
        self.imagepaths = util.read_pickle(self.FE.pytrochnormalimagepath)
        self.vectors = util.read_pickle(self.FE.pytorchnormalfepath)

    def get_similar_images_normal(self,img):
        featurevector = self.FE.extract(img)
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(self.vectors)
        distances, indices = neighbors.kneighbors([featurevector])

        files = []
        for ind in indices[0]:
            file = self.imagepaths[ind]
            files.append(file)
        return files



class SearchImageNormaltensorflow:
    def __init__(self,config):
        self.FE = NoramlfeatureextractorTensorflow(config)
        self.vectorsten = util.read_pickle(self.FE.tensnormalfepath)
        self.imagesten= util.read_pickle(self.FE.tensnormalimagepath)

    def get_similar_images_normal(self,img):
        featurevector = self.FE.extract(img)
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(self.vectorsten)
        distances, indices = neighbors.kneighbors([featurevector])

        files = []
        for ind in indices[0]:
            file = self.imagesten[ind]
            files.append(file)
        return files


def normalmain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', "-c", type=str, default="config/config.yaml", help='ROOT/config/config.yaml')
    parser.add_argument('--params', "-p", type=str, default="params.yaml", help='ROOT/params.yaml')
    opt = parser.parse_args()

    FEpt = NoramlfeatureextractorPytroch(**vars(opt))
    FEpt.get_features_normal()
    FEten = NoramlfeatureextractorTensorflow(next(iter(vars(opt).values())))
    FEten.get_features_normal()

if __name__ == '__main__':
    normalmain()
