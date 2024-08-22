import requests
import shutil
from bs4 import BeautifulSoup
import os
import threading
from PIL import Image
import time
# pip install lxml


def downloadImage(gene, ID, tissueSubBlock):
    Image.MAX_IMAGE_PIXELS = 971441920
    # Download ISH Image

    while True:
        img = requests.get(f"http://api.brain-map.org/api/v2/image_download/{ID}", stream=True)
        try:
            if img.status_code == 200:
                with open(f"{tissueSubBlock}/{tissueSubBlock}_{ID}_{gene}_ISH.jpg", 'wb') as f:
                    img.raw.decode_content = True
                    shutil.copyfileobj(img.raw, f)
                try:
                    i = Image.open(f"{tissueSubBlock}/{tissueSubBlock}_{ID}_{gene}_ISH.jpg")
                    break
                except Exception as e:
                    print(e)
                    print(f"{tissueSubBlock}/{tissueSubBlock}_{ID}_{gene}_ISH.jpg")
                    time.sleep(1)
                    continue
            else:
                print(img.status_code)

        except Exception as e:
            print(e)




    # Download corresponding annotated image
    while True:
        img = requests.get(f"http://api.brain-map.org/api/v2/image_download/{ID}?view=tumor_feature_annotation", stream=True)
        try:
            if img.status_code == 200:
                with open(f"{tissueSubBlock}/{tissueSubBlock}_{ID}_{gene}_annotated.jpg", 'wb') as f:
                    img.raw.decode_content = True
                    shutil.copyfileobj(img.raw, f)
                try:
                    i = Image.open(f"{tissueSubBlock}/{tissueSubBlock}_{ID}_{gene}_annotated.jpg")
                    break
                except Exception as e:
                    print(e)
                    print(f"{tissueSubBlock}/{tissueSubBlock}_{ID}_{gene}_annotated.jpg")
                    # downloadImage(gene, ID, tissueSubBlock)
                    continue
            else:
                print(img.status_code)

        except Exception as e:
            print(e)


    print(gene)



def requestImage(tissueSubBlock):

    try:
        os.mkdir(f"{tissueSubBlock}")
        print("made")
    except:
        print("sub block directory already exists")
        return

    # Get ID For Sub Block
    getID = requests.get(f"http://api.brain-map.org/api/v2/data/query.xml?criteria=model::SectionDataSet,rma::criteria,specimen[external_specimen_name$eq'{tissueSubBlock}'],treatments[name$eq'ISH'],rma::include,genes,sub_images", stream=True)
    if getID.status_code == 200:
        Bs_data = BeautifulSoup(getID.text, "xml")
        images = Bs_data.findAll("section-data-set")
        threads = []

        for image in images:
            try:
                gene = image.findNext("acronym").text
                print(gene)
                ID = image.find("sub-image").find("id").text
                with open('filenames.txt', 'a') as file:
                    file.write(f"{tissueSubBlock}/{tissueSubBlock}_{ID}_{gene}\n")
                t = threading.Thread(target=downloadImage, args=(gene, ID, tissueSubBlock))
                threads.append(t)
            except:
                pass

        # Start all threads
        for x in threads:
            x.start()

        # Wait for all of them to finish
        for x in threads:
            x.join()

    else:
        print(getID.status_code)
        return
    
requestImage("W31-1-1-E.03")