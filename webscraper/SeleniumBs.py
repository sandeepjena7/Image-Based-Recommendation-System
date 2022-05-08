from DeepImageSearch.utils.allutils import util
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
import os
import pickle
from tqdm import tqdm
import argparse
from colorama import Fore
import warnings
warnings.simplefilter("ignore")


class MyntraScraper:
    """
    "User-Agent-About-Our-System": Chrome/101.0.4951.41   Windows10 64-bit operating system"
    """
    def __init__(self,myntra,config):
        self.df = pd.DataFrame(columns=["Imagename","Category","BrandName","Price","Ratting",'NoOfPurchasing',"WebsiteProductLink","ImageLink"])
        self._loadparams(myntra)
        config = util.read_yaml(config)
        artifact_dir = config['artifacts']['artifactdir']
        imagedir = config['artifacts']["image_dir"]
        train_dir = config['artifacts']["train_dir"]
        val_dir = config['artifacts']["val_dir"]
        metadata_dir = config['artifacts']["meta_data_dir"]
        dataframefilesname = config['artifacts']["dataframepkl"]
        self.dataframefilepath = os.path.join(artifact_dir,metadata_dir,dataframefilesname)

        os.makedirs(artifact_dir,exist_ok=True)
        os.makedirs(os.path.join(artifact_dir,imagedir),exist_ok=True)
        os.makedirs(os.path.join(artifact_dir,imagedir,train_dir),exist_ok=True)
        os.makedirs(os.path.join(artifact_dir,imagedir,val_dir),exist_ok=True)
        os.makedirs(os.path.join(artifact_dir,metadata_dir),exist_ok=True)

        self.option = webdriver.ChromeOptions()
        self.option.add_argument('--disable-gpu')
        self.option.add_argument('--no-sandbox')
        self.option.add_argument("disable-dev-shm-usage")


    def _loadparams(self,myntraScraperyamlfile):
        myntraconfig = util.read_yaml(myntraScraperyamlfile)
        self.webdriverpath = myntraconfig['webscrapers']['webdriverpath']

        self.myntralinks = myntraconfig['myntraparams']['myntralinks']
        self.sublinks = myntraconfig['myntraparams']['sublinks']
        self.product = myntraconfig['myntraparams']["product"]
        self.ratings = myntraconfig['myntraparams']['ratings']
        self.imglink = myntraconfig['myntraparams']['imglink']
        self.price = myntraconfig['myntraparams']['price']
        self.productbuywebsite = myntraconfig['myntraparams']['productbuywebsite']
        self.brandname = myntraconfig['myntraparams']['brandname']

        self.pagetotalsearch = myntraconfig['user']['totalpageonecategory']
        self.categorys = myntraconfig['user']['categorys']

    def start(self):
        since = time.time()
        imagenameadd = 1000
        index = 0
        driver = webdriver.Chrome(executable_path=self.webdriverpath, options=self.option)

        for category in tqdm(self.categorys,ncols = 100,desc ="webscraping process start category",colour='MAGENTA'):
            for pagenumber in range(self.pagetotalsearch):
                linksearch = f"{self.myntralinks }/{category}{self.sublinks}{pagenumber + 1}"

                try:
                    driver.get(linksearch)
                    driver.implicitly_wait(1)
                    y = 1000
                    # https://stackoverflow.com/questions/30942041/slow-scrolling-down-the-page-using-selenium
                    for timer in range(0, 7):
                        driver.execute_script("window.scrollTo(0, " + str(y) + ")")
                        y += 1000
                        time.sleep(3)  # if internate is slow then sleep time to change greater then 3 so that when scrolling
                        # it loads all images  so that we scraping all images  links
                    htmlpage = bs(driver.page_source, "html.parser")

                except:
                    driver.quit()
                    driver = webdriver.Chrome(executable_path=self.webdriverpath, options=self.option)
                    continue

                else:
                    products = htmlpage.find_all(self.product[0],{self.product[1]:self.product[2]})

                    for product in products:

                        ratingbuys = product.find_all(self.ratings[0],{self.ratings[1]:self.ratings[2]})
                        imglink = product.find_all(self.imglink[0],{self.imglink[1]:self.imglink[2]})
                        price = product.find_all(self.price[0],{self.price[1]:self.price[2]})
                        linkbuyproduct = product.find_all(self.productbuywebsite[0],{self.productbuywebsite[1]:self.productbuywebsite[2]})
                        brandname = product.find_all(self.brandname[0],{self.brandname[1]:self.brandname[2]})
                        if len(imglink) & len(price) & len(linkbuyproduct) & len(brandname):
                            """ in one product this three should be present if not present then we don't added our dataframe """

                            if len(ratingbuys):
                                ratingtotalbuys = ratingbuys[0].text.split("|")
                                self.df.loc[index, "Ratting"] = ratingtotalbuys[0]
                                self.df.loc[index, "NoOfPurchasing"] = ratingtotalbuys[1]

                            else:
                                self.df.loc[index, "Ratting"] = 0
                                self.df.loc[index, "NoOfPurchasing"] = 0

                            self.df.loc[index, "Imagename"] = f"{imagenameadd}.jpg"
                            self.df.loc[index, "Category"] = category
                            self.df.loc[index, "ImageLink"] = imglink[0]['src']
                            self.df.loc[index, "Price"] = price[0].text.replace(" ", "")
                            self.df.loc[index, "WebsiteProductLink"] = linkbuyproduct[0]['href']
                            self.df.loc[index,"BrandName"] = brandname[0].text

                            imagenameadd += 1
                            index += 1

        driver.quit()
        time_elapsed = time.time() - since
        print(f"Total Time Taken To complete {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        dictionary = self.df.to_dict('dict')
        filehandler = open(self.dataframefilepath ,"wb")
        pickle.dump(dictionary, filehandler)
        filehandler.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--myntra',"-m", type=str, default="config/myntra.yaml", help='ROOT/config/myntra.yaml')
    parser.add_argument('--config',"-c", type=str, default="config/config.yaml", help='ROOT/config/config.yaml')

    opt = parser.parse_args()
    scraper = MyntraScraper(**vars(opt))
    scraper.start()

if __name__ == '__main__':
    main()