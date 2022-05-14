from DeepImageSearch.Normalfeatureextractor import SearchImageNormalPytoch,SearchImageNormaltensorflow
from DeepImageSearch.Ccbrfeatureextractor import SearchImagePytorch,SearchImageTensorflow
from DeepImageSearch.utils.allutils import util
import pandas as pd
import os
import pickle
import path



class RUN:
    def __init__(self,config,params):
        self.config = config
        self.params = params

        config = util.read_yaml(config)
        artifact_dir = config["artifacts"]["artifactdir"]
        metadata_dir = config["artifacts"]["meta_data_dir"]
        analysisdataframe = config['artifacts']['analysisdataframe']
        imgsearch = config['artifacts']["PytorchCCBRImagefile"]
        imagespkl = os.path.join(artifact_dir,metadata_dir,imgsearch)
        data_path = os.path.join(artifact_dir,metadata_dir,analysisdataframe)
        dict_ = util.read_pickle(data_path)
        self.imagepath = util.read_pickle(imagespkl)
        self.df = pd.DataFrame(dict_)

        

    def search(self,Framework,Technique,image,usedcase="streamlite"):
        if os.path.isfile(image):


            if Framework in "Pytorch":
                if Technique in "CCBR": 
                    imagespath = SearchImagePytorch(self.config,self.params).get_similar_images(image)
                elif Technique in "Pre-trained":
                    imagespath = SearchImageNormalPytoch(self.config,self.params).get_similar_images_normal(image)
                else:
                    return None

            elif Framework in "Tensorflow":
                if Technique in 'CCBR':
                    imagespath = SearchImageTensorflow(self.config).get_similar_images(image)
                elif Technique in 'Pre-trained':
                    imagespath = SearchImageNormaltensorflow(self.config).get_similar_images_normal(image)
                else: return None
            else: return None
            
            if usedcase in 'streamlite' and len(imagespath):
                imgname = list(map(lambda x:os.path.basename(x), imagespath))
                toprecomenddf = self.df[self.df["Imagename"].isin(imgname)]
                toprecomenddf = toprecomenddf.reset_index().drop(['index',"ImageLink"],axis=1)
                for i,imgpath in enumerate(imagespath):
                    toprecomenddf.loc[i,"Imagename"] = imgpath

                toprecomendcategory = toprecomenddf['Category'].mode().values[0]
                top5rattingdf = self.df.groupby(['Category']).get_group(toprecomendcategory).sort_values('Ratting',ascending=False).head(5)
                top5rattingimgname = list(top5rattingdf['Imagename'])
                top5rattingimgpath = [ospath for ospath in self.imagepath[toprecomendcategory] for name in top5rattingimgname if os.path.basename(ospath) in name ] # imagepath[toprecomendcategory] is used to reduce time complexity
                top5rattingdf = top5rattingdf.reset_index().drop(['index',"ImageLink"],axis=1)
                for i ,imgspath in enumerate(top5rattingimgpath):
                    top5rattingdf.loc[i,'Imagename'] = imgspath
                    

                value = {'recomnddf':toprecomenddf.to_dict('list'),'ratting5':top5rattingdf.to_dict('list')}
                return value
                
            elif usedcase == 'api':
                encodeimage = []

                for i,imgpath in enumerate(imagespath):
                    encodeimage.append({f"image{i}":util.encode_base64(imgpath)})
                
                return encodeimage

        

    



if __name__ == '__main__':
    j = RUN('config/config.yaml','params.yaml')
    j.search('Tensorflow','CCBR',"recommendation.png")
    