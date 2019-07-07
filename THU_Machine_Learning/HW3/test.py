import json
import ijson

# source: https://stackoverflow.com/a/3229493
def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def name_from_key(key):
    parts = key.split('_')
    parts = map(lambda p: p.capitalize() + ('.' if len(p) == 1 else ''), parts)
    return ' '.join(parts)

key = ''
name = ''

inst_to_id = {}
results = {}

keep_next_org = False

# FILENAME = './test.json'
# FILENAME = './pubs_train.json'
FILENAME = './pubs_validate.json'

parser = ijson.parse(open(FILENAME))

for prefix, event, value in parser:
    if (prefix, event) == ('', 'map_key'):
        key = value
        name = name_from_key(key)
        inst_to_id = {}

    elif (prefix, event, value) == (f'{key}.item.authors.item.name', 'string', name):
        keep_next_org = True

    elif (prefix, event) == (f'{key}.item.authors.item.org', 'string') and keep_next_org:
        org = value
        keep_next_org = False

    elif (prefix, event) == (f'{key}.item.id', 'string'):
        id = value
        if org not in inst_to_id: inst_to_id[org] = []
        inst_to_id[org].append(id)

    elif (prefix, event) == (key, 'end_array'):
        results[key] = []
        for inst, id_list in inst_to_id.items():
            results[key].append(id_list)

print(json.dumps(results, indent=4))
