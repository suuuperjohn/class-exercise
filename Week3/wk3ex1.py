import json

my_dict = {
  "name": "John",
  "age": 30,
  "married": True,
  "children": [

    {"name": "Jenny", "age": 4},

    {"name": "Billy", "age": 2}

  ],
  "pets": None,
  "cars": [
    {"model": "BMW 230", "mpg": 27.5},
    {"model": "Ford Edge", "mpg": 24.1}
  ]
}

print(json.dumps(my_dict, indent=4))

print(my_dict['name'])

for car in my_dict['cars']:
    if car['mpg'] > 25:
        print(car['model'])


if my_dict['married'] == True:
    for child in my_dict['children']:
        print(child['age'])


for child in my_dict["children"]:
  if child['name'] == 'Jenny':
      child ['age'] = 5

print(json.dumps(my_dict, indent=4))

with open('john.json', 'w') as f:
  json.dump(my_dict, f)


import json

with open('john.json') as f:
  my_dict = json.load(f)
  print(json.dumps(my_dict, indent = 4))
  del my_dict['age']
  with open('john2.json', 'w') as f:
     json.dump(my_dict, f, indent=4)


with open('john.json', 'w') as f:
  json.dump(my_dict, f, indent=4)