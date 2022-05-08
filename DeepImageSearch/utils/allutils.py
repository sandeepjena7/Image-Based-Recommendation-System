import yaml
import sys
import os
from pathlib import Path
import glob
import pickle
import logging.config
import logging


class SetUpLogging():
    def __init__(self,loggingfilepath):
        self.default_config = os.path.join(os.path.dirname(
            os.path.abspath('__file__')),loggingfilepath)

    def setup_logging(self, default_level=logging.info):
        path = self.default_config
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                logging.captureWarnings(True)
        else:
            logging.basicConfig(level=default_level)


class util:
    @staticmethod
    def read_yaml(filename)  :
        try:
            with open(filename) as yaml_files:
                content = yaml.safe_load(yaml_files)
            return content
        except OSError:
            print(f"files is not open {filename}")
            sys.exit()
        except Exception as e:
            print(e)

    @staticmethod
    def dump_pickle(filename,data):
        try:
            filehandle = open(filename,"wb")
            pickle.dump(data,filehandle)
            filehandle.close()
        except OSError:
            print(f"pickle file is not dump our data")
            sys.exit()
        except Exception as e:
            print(e)

    @staticmethod
    def read_pickle(filename):
        try:
            with open(filename,"rb") as file:
                data = pickle.load(file)
            return data
        except OSError:
            print(f"dump file pat or data some error occured")
            sys.exit()
        except Exception as e:
            print(e)




class LoadImagesNormal:

    def __init__(self, path, custom=None):

        p = str(Path(path).resolve())

        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))

        elif os.path.isdir(p):
            if custom is None:
                files = sorted(glob.glob(os.path.join(p, '*.*')))
            else:
                subfolder = [dir_.path for dir_ in os.scandir(p) if dir_.is_dir()]
                child = [dir_.path for child in subfolder for dir_ in os.scandir(child) if dir_.is_dir()]
                listoffiles = [sorted(glob.glob(os.path.join(dir_, '*.*'))) for dir_ in child]
                files = [file for listfile in listoffiles for file in listfile]
                assert len(files), " file is not present in sub dir"

        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        self.nf = len(files)  # number of files
        self.files = files

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration

        path = self.files[self.count]
        self.count += 1

        return path

    def __len__(self):
        return self.nf


class LoadImagesCCBR:
    def __init__(self, path):
        p = str(Path(path).resolve())
        if os.path.isdir(p):
            subfolder = [dir_.path for dir_ in os.scandir(p) if dir_.is_dir()]

            child = [dir_.path for child in subfolder for dir_ in os.scandir(child) if dir_.is_dir()]
            listoffiles = [sorted(glob.glob(os.path.join(dir_, '*.*'))) for dir_ in child]
            files = []
            self.uniquecategory = []

            for listfiles in listoffiles:
                for file in listfiles:
                    categories = os.path.basename(os.path.dirname(file))
                    if categories not in self.uniquecategory:
                        self.uniquecategory.append(categories)

                    files.append({categories: file})

        self.nf = len(files)
        self.files = files

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):

        if self.count == self.nf:
            raise StopIteration

        path = self.files[self.count]
        self.count += 1

        name = list(path.keys())[0]
        filepath = list(path.values())[0]

        return name, filepath

    def __len__(self):
        return self.nf
