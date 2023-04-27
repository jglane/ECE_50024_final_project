from bs4 import BeautifulSoup
import requests
import re
import os
import sys
import imghdr

def scrape_images(search_string, dir, training):
    os.mkdir(dir)
    
    PAGES = [1, 2] if training else [3, 4]
    i = 0
    for page_num in PAGES:
        url = f'https://www.flickr.com/search/?text={search_string}&per_page=501&page={page_num}'
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')

        div = soup.find('div', class_='main search-photos-results')
        imgs_src = re.findall('src="(.+?)"', str(div))
        
        for img_src in imgs_src:
            img_src = img_src[:-5] + 'e.jpg'
            img = requests.get(f'https:{img_src}')
            with open(f'{dir}/{i}.jpg', 'wb') as f:
                f.write(img.content)
            
            if imghdr.what(f'{dir}/{i}.jpg') != 'jpeg':
                os.remove(f'{dir}/{i}.jpg')
            else:
                i += 1

search_stringA = sys.argv[1]
search_stringB = sys.argv[2]
image_set = f'{search_stringA.replace(" ", "_")}2{search_stringB.replace(" ", "_")}'
os.mkdir(f'data/{image_set}')

scrape_images(search_stringA, f'data/{image_set}/trainA', training=True)
scrape_images(search_stringB, f'data/{image_set}/trainB', training=True)

scrape_images(search_stringA, f'data/{image_set}/testA', training=False)
scrape_images(search_stringB, f'data/{image_set}/testB', training=False)

os.mkdir(f'tests/{image_set}')
# add test images to this directory