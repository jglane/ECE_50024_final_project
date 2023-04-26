from bs4 import BeautifulSoup
import requests
import re
import os
import sys
import imghdr

def scrape_images(search_string, dir):
    os.mkdir(dir)
    
    NUM_PAGES = 2
    i = 0
    for page_num in range(NUM_PAGES):
        url = f'https://www.flickr.com/search/?text={search_string}&per_page=501&page={page_num + 1}'
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
image_set = f'{search_stringA}2{search_stringB}'
os.mkdir(f'data/{image_set}')

scrape_images(search_stringA, f'data/{image_set}/trainA')
scrape_images(search_stringB, f'data/{image_set}/trainB')

# cmd: /path/to/python scraper.py search_stringA search_stringB
# example: /home/jglane/.conda/envs/cent7/2020.11-py38/my_tf_env/bin/python scraper.py tiger lion