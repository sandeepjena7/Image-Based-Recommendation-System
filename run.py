from DeepImageSearch.Normalfeatureextractor import SearchImageNormalPytoch,SearchImageNormaltensorflow
from DeepImageSearch.Ccbrfeatureextractor import SearchImagePytorch,SearchImageTensorflow
from DeepImageSearch.utils.allutils import util
import pandas as pd
import os

class RUN:
    def __init__(self,config,params):
        self.pyccbr = SearchImagePytorch(config,params)
        self.tfcbbr = SearchImageTensorflow(config)
        self.pynormal = SearchImageNormalPytoch(config,params)
        self.tfnormal = SearchImageNormaltensorflow(config)
        config = util.read_yaml(config)
        artifact_dir = config["artifacts"]["artifactdir"]
        metadata_dir = config["artifacts"]["meta_data_dir"]
        analysisdataframe = config['artifacts']['analysisdataframe']
        data_path = os.path.join(artifact_dir,metadata_dir,analysisdataframe)
        dict_ = util.load_pickle(data_path)
        self.df = pd.DataFrame(dict_)

        

    def search(self,Framework,Technique,image):

        if Framework == "Pytroch":
            if Technique == "ccbr": 
                imagespath = self.pyccbr.get_similar_images(image)
            else:
                imagespath = self.pynormal.get_similar_images_normal(image)

        else:
            if Technique == 'ccbr':
                imagespath = self.tfccbr.get_similar_images(image)
            else:
                imagespath = self.tfnormal.get_similar_images_normal(image)

        data = {keys: [] for keys in self.df.columns}
        for path in imagespath:
            
            data['Imagename'] 
    



if __name__ == '__main__':
    j = RUN('config/config.yaml','params.yaml')
    