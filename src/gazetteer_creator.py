import os
import json
import re

def find_interwiki_links(wikipages, wiki):
    content = wiki['text']
    titles = re.findall(r'href="([^"]+)"', content)

    titles = [title.replace('%20', ' ') for title in titles]
    for i in range(len(titles)):
        if '%23' in titles[i]:
            titles[i] = titles[i][:titles[i].index('%23')]
    titles = [title.replace('%27', '\'') for title in titles]
    titles = [title.replace('%28', '(') for title in titles]
    titles = [title.replace('%29', ')') for title in titles]
    titles = [title.replace('%2C', ',') for title in titles]
    titles = [title.replace('%C3A9', 'é') for title in titles]
    titles = [title.replace('%E2%80%93', '-') for title in titles]
    titles = [title.replace('%C3%A3', 'ã') for title in titles]
    titles = [title.replace('%C3%A7', 'ç') for title in titles]
    titles = [title.replace('%26amp%3B', '&') for title in titles]

    for title in titles:
        if title == '':
            titles.remove(title)

    titles = list(set(titles))
    titles = [t for t in titles if t]

    anchors = re.findall(r'&gt;([^&]+)&lt;', content)
    gazetteer = titles + anchors
    return gazetteer

def create_gazetteer(PATHWIKI):
    wikipages = {}
    count = 0

    for file in os.listdir(PATHWIKI):
        if count % 1000 == 0 and count != 0:
            print(count)

        with open(PATHWIKI + '/' + file) as f:
            wikipages[file.split('.')[0]] = json.load(f)

        count += 1

    gazetteer = []
    for key in wikipages.keys():
        wiki = wikipages[key]
        gazetteer += find_interwiki_links(wikipages, wiki)

    gazetteer = list(set(gazetteer))

    to_remove = []
    for i in range(len(gazetteer)):
        if gazetteer[i] == '':
            to_remove.append(i)

    for i in reversed(to_remove):
        del gazetteer[i]

    gazetteer = [word for word in gazetteer if word[0].isupper()]
    gazetteer = [word for word in gazetteer if len(word) > 1]
    gazetteer = [re.sub(r'[^\w]', '', word) for word in gazetteer]

    gazetteer = list(set(gazetteer))

    print('Gazetteer size: ', len(gazetteer))

    return gazetteer

