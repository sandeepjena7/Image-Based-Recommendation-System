import argparse
import os
from image_resize import letterbox
from DeepImageSearch.utils.allutils import util
import cv2
from tqdm import tqdm

class Resize(object):
    def __init__(self,image_path,output_path):
        config = util.read_yaml(config)
        artifact_dir = config['artifact']["artifactdir"]
        imgdir = config['artifact']['image_dir']
        raw = config['artifact']['raw']
        processed = config['artifact']['preprocessed']
        self.image_path = os.path.join(artifact_dir,imgdir,raw)
        self.output_path = os.path.join(artifact_dir,imgdir,processed)

    def convert(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        folders = ['train','val']
        for ele in folders:
            for r, d, f in tqdm(os.walk(os.path.join(self.image_path,ele)),colour="GREEN",desc=f"Working with {ele} folder"):
                for dir in d:
                        dir_path = os.path.join(self.image_path,ele,dir)
                        for img in tqdm(os.listdir(dir_path),colour="MAGENTA",desc=f"Resizing for {dir}"):
                            dst_dir = os.path.join(self.output_path,ele,dir)
                            if not os.path.exists(dst_dir):
                                os.makedirs(dst_dir)
                            inp_path = os.path.join(dir_path,img)
                            out_path = os.path.join(dst_dir,img)
                            im = cv2.imread(inp_path)
                            out_img = letterbox(im)
                            cv2.imwrite(out_path,out_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config','-c', type=str, default='config/config.yaml', help='ROOT  config/config.yaml')

    opt = parser.parse_args()
    resize = Resize(**vars(opt))
    resize.convert()


if __name__ == "__main__":
    main()
