#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 03:35:08 2019

@author: zhangnaifu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:10:35 2019

@author: zhangnaifu
"""

import json
import ijson
import string

def name_from_key(key):
    parts = key.split('_')
    parts = map(lambda p: p.capitalize() + ('.' if len(p) == 1 else ''), parts)
    return ' '.join(parts)

translator = str.maketrans('', '', string.punctuation)

key = ''
name = ''

org_dict = {}
venue_dict={}
results = {}

keep_next_org = False

# FILENAME = './test.json'
# FILENAME = './pubs_train.json'
FILENAME = 'data/pubs_test.json'

parser = ijson.parse(open(FILENAME))

for prefix, event, value in parser:
    
    if (prefix, event) == ('', 'map_key'):
        print(value)
        key = value
        name = name_from_key(key)
        org_dict = {}
        venue_dict={}

    elif (prefix, event, value) == ('{}.item.authors.item.name'.format(key), 'string', name):
        keep_next_org = True

    elif (prefix, event) == ('{}.item.authors.item.org'.format(key), 'string') and keep_next_org:
        if value != "":
            org = value
        keep_next_org = False
        
    elif (prefix, event) == ('{}.item.venue'.format(key), 'string'):
        if value !="":
            venue = value.translate(translator)
            venue = venue.lower()

    elif (prefix, event) == ('{}.item.id'.format(key), 'string'):
        id_pub = value
        if org not in org_dict: 
            org_dict[org] = []
            org_dict[org].append(venue)
        org_dict[org].append(id_pub)
        
        if venue not in venue_dict: 
            venue_dict[venue] = []
        venue_dict[venue].append(id_pub)
    

    elif (prefix, event) == (key, 'end_array'):
        results[key] = []
        for inst, id_list in org_dict.items():
            pubs_list = id_list[1:] + venue_dict[id_list[0]]
            pubs_list = list(set(pubs_list))
            results[key].append(pubs_list)
            

with open('result_test_1.json', 'w') as f:
    json.dump(results, f)
