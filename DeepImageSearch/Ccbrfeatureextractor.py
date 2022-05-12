from DeepImageSearch.PYtorch import Net
from DeepImageSearch.utils.allutils import util,LoadImagesCCBR
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from torch import nn
import torch
from PIL import Image
import torchvision.transforms as  T
from numpy.linalg import norm
from tqdm import tqdm
import pickle
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse

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


class FeatureExtractorPytorch:
    def __init__(self,config,params):
        config = util.read_yaml(config)
        params = util.read_yaml(params)
        artifact_dir = config["artifacts"]["artifactdir"]
        image_dir = config["artifacts"]["image_dir"]
        preprocessed = config['artifacts']['preprocessed']
        metadata_dir = config["artifacts"]["meta_data_dir"]
        pytccbrfilename = config["artifacts"]["PytorchCCBRFEfile"]
        pytorchccbrimagefile = config["artifacts"]["PytorchCCBRImagefile"]
        trainmodelpytorch = config['artifacts']['pytorchmodel']
        output_class = params['class']

        self.imagedir = os.path.join(artifact_dir,image_dir,preprocessed)
        self.metadatapytorchimagenamesfilename = os.path.join(artifact_dir,metadata_dir,pytorchccbrimagefile)
        self.metadatapytrochccbrfeatures = os.path.join(artifact_dir,metadata_dir,pytccbrfilename)
        pretrained_model_path = os.path.join(artifact_dir,metadata_dir,trainmodelpytorch)
        self.model = Net(inputchannel=3, output_class=output_class)
        self.model.load_state_dict(torch.load(pretrained_model_path))
        _PytorchParameters.set_parameter_requires_grad(self.model)

        self.modelfeaturevecotr = Net(inputchannel=3, output_class=output_class) # used paramsconfig file
        self.modelfeaturevecotr.load_state_dict(torch.load(pretrained_model_path))
        _PytorchParameters.set_parameter_requires_grad(self.modelfeaturevecotr)
        # print(_PytorchParameters.show_params_learn(self.modelfeaturevecotr)) # when enver conform comment it out this line
        self.modelfeaturevecotr.fc =  nn.Sequential(
                                    nn.AdaptiveAvgPool2d(1)
                                    ,nn.Flatten())


    @classmethod
    def Transform(cls,pillowimg):
        transform = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the data base on imagenet datasets
            ])
        img = transform(pillowimg)
        return img

    def extract(self,img):
        img = Image.open(img).convert("RGB") # some of image not rgb so i fact this issue i found easy to overcome this is here conver image
        img = self.Transform(img)
        if len(img.shape) == 3:
            img = img[None]

        feature = self.modelfeaturevecotr(img)[0] # please check shape of feature or [0] when debuggning
        return feature/norm(feature) # unit vector retrun through linear algebra

    def get_features(self):
        dataset = LoadImagesCCBR(self.imagedir)
        features = {keys:[] for keys in dataset.uniquecategory}
        filenames = {keys:[] for keys in dataset.uniquecategory}

        for category,path in tqdm(dataset,ncols = 100,colour='YELLOW'):
            try:
                feat = self.extract(path).numpy()
                features[category].append(feat)
                filenames[category].append(path)
            except :
                features[category].append(None)
                filenames[category].append(path)
                continue

        util.dump_pickle(self.metadatapytrochccbrfeatures,features)
        util.dump_pickle(self.metadatapytorchimagenamesfilename,filenames)


class SearchImagePytorch:
    def __init__(self,config,params):
        self.FE = FeatureExtractorPytorch(config,params)
        config = util.read_yaml(config)
        artifact_dir = config["artifacts"]["artifactdir"]
        image_dir = config["artifacts"]["image_dir"]
        preprocessed = config['artifacts']['preprocessed']
        traindir = config["artifacts"]["train_dir"]

        dir_ = os.path.join(artifact_dir,image_dir,preprocessed,traindir)
        self.classes = os.listdir(dir_)

        self.dictcatfilename = util.read_pickle(self.FE.metadatapytorchimagenamesfilename)
        self.dictcatvector = util.read_pickle(self.FE.metadatapytrochccbrfeatures)
        



    def PredictClass(self,img):
        img = Image.open(img).convert("RGB")  # some of image not rgb so i fact this issue i found easy to overcome this is here conver image
        img = self.FE.Transform(img)
        if len(img.shape) == 3:
            img = img[None]

        self.FE.model.eval()
        output = self.FE.model(img)
        index = output.data.numpy().argmax()
        return index

    def get_similar_images(self,img):
        index = self.PredictClass(img)
        featurevector = self.FE.extract(img).numpy()
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(self.dictcatvector[self.classes[index]])
        
        distances, indices = neighbors.kneighbors([featurevector])
        files = [] 
        for ind in indices[0]:
            file = self.dictcatfilename[self.classes[index]][ind]
            files.append(file)

        return files[1:]



class FeatureExtractorTensorflow:
    def __init__(self,config):
        config = util.read_yaml(config)
        artifact_dir = config["artifacts"]["artifactdir"]
        image_dir = config["artifacts"]["image_dir"]
        preprocessed = config['artifacts']['preprocessed']
        metadata_dir = config["artifacts"]["meta_data_dir"]
        tensccbrfilename = config["artifacts"]["TensorFlowCCBRfile"]
        tenchccbrimagefile = config["artifacts"]["TesnsorFlowCCBRImagefile"]
        trainmodelten = config['artifacts']['tensoflowmodel']

        self.imagedir = os.path.join(artifact_dir, image_dir,preprocessed)
        self.metadataptenorimagenamesfilename = os.path.join(artifact_dir, metadata_dir, tenchccbrimagefile)
        self.metadatatensorccbrfeatures = os.path.join(artifact_dir, metadata_dir, tensccbrfilename)
        tens_model_path = os.path.join(artifact_dir, metadata_dir, trainmodelten)

        self.model = tf.keras.models.load_model(tens_model_path)
        self.model.trainable = False

        self.modelfeaturevecotr = tf.keras.Sequential(self.model.layers[:-8]) # change this as for your model only remove flatten layers if dropout here that also removes
        self.modelfeaturevecotr.add(GlobalAveragePooling2D())

    def extract(self,img):
        img = image.load_img(img,target_size=(224, 224)) # change your size as per input Model
        img = image.img_to_array(img)
        img /= 255.0 # becaue we do that when we train our model there is we do rescale operation so that there we have used this operation
        if len(img.shape) == 3:
            img = img[None] # simple expand dims

        feature = self.modelfeaturevecotr.predict(img)[0] #used predict
        return feature/norm(feature)

    def get_features(self):
        dataset = LoadImagesCCBR(self.imagedir)
        features = {keys: [] for keys in dataset.uniquecategory}
        filenames = {keys: [] for keys in dataset.uniquecategory}

        for category,path in tqdm(dataset,ncols = 100,colour='CYAN'):
            try:
                feat = self.extract(path)
                features[category].append(feat)
                filenames[category].append(path)

            except :
                features[category].append(None)
                filenames[category].append(path)
                continue

        util.dump_pickle(self.metadatatensorccbrfeatures ,features)
        util.dump_pickle(self.metadataptenorimagenamesfilename,filenames)

class SearchImageTensorflow:
    def __init__(self,config):
        self.FE = FeatureExtractorTensorflow(config)
        config = util.read_yaml(config)
        artifact_dir = config["artifacts"]["artifactdir"]
        image_dir = config["artifacts"]["image_dir"]
        preprocessed = config['artifacts']['preprocessed']
        traindir = config["artifacts"]["train_dir"]

        self.imagedir = os.path.join(artifact_dir, image_dir,preprocessed)
        self.dictcatfilename = util.read_pickle(self.FE.metadataptenorimagenamesfilename)
        self.dictcatvector = util.read_pickle(self.FE.metadatatensorccbrfeatures)

        dir_ = os.path.join(artifact_dir,image_dir,preprocessed,traindir)
        self.classes = os.listdir(dir_)
        

    def PredictClass(self,img):
        img = image.load_img(img, target_size=(224, 224))  # change your size as per input Model
        img = image.img_to_array(img)
        img /= 255.0  # becaue we do that when we train our model there is we do rescale operation so that there we have used this operation
        if len(img.shape) == 3:
            img = img[None]  # simple expand dims

        output = self.FE.model(img) # used predict  normalfeatures
        index = np.argmax(output.numpy()[0],axis=0)
        return index

    def get_similar_images(self,img):
        index = self.PredictClass(img)
        featurevector = self.FE.extract(img)
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(self.dictcatvector[self.classes[index]])

        distances, indices = neighbors.kneighbors([featurevector])
        files = [] 
        for ind in indices[0]:
            file = self.dictcatfilename[self.classes[index]][ind]
            files.append(file)
        return files[1:]


def ccbrmain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', "-c", type=str, default="config/config.yaml", help='ROOT/config/config.yaml')
    parser.add_argument('--params', "-p", type=str, default="params.yaml", help='ROOT/params.yaml')
    opt = parser.parse_args()

    FEpt = FeatureExtractorPytorch(**vars(opt))
    FEpt.get_features()
    FEten = FeatureExtractorTensorflow(next(iter(vars(opt).values())))
    FEten.get_features()

if __name__ == '__main__':
    ccbrmain()





