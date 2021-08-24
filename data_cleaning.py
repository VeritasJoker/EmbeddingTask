import json

reviews = [] # empty array of dicts to store data
with open('apps.json') as file:  # open json file
    for reviewjson in file: # read through lines
        review = json.loads(reviewjson) # load json object
        reviews.append(review) # append dict to array of dict

with open('apps_cleaned.json','w') as outfile: # open new json file
    for review in reviews:
        if review['overall'] != 3: # just keep high and low
            json.dump({key: review[key] for key in ['overall','reviewText']},outfile) # write json object
            outfile.write("\n") # new line


