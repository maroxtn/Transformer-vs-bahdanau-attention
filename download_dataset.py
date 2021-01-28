import wget 
import zipfile
import os
import shutil



url = "https://github.com/maroxtn/IWSLT-BACKUP/archive/main.zip" 
wget.download(url)


with zipfile.ZipFile("IWSLT-BACKUP-main.zip", 'r') as zip_ref:
    zip_ref.extractall("tmp")


os.mkdir("data")

shutil.move("tmp/IWSLT-BACKUP-main/iwslt", "data/") #Move directory
shutil.rmtree("tmp")

os.remove("IWSLT-BACKUP-main.zip")
