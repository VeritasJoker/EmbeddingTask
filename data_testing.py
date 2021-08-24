import json

reviews = [] # empty array of dicts to store data
with open('apps.json') as file:  # open json file
    for reviewjson in file: # read through lines
        review = json.loads(reviewjson) # load json object
        reviews.append(review) # append dict to array of dict

# for data testing
ratings = []
high = 0
low = 0
for review in reviews:
    if review['overall'] >= 4:
        high += 1
    if review['overall'] <= 2:
        low += 1
print(high, low, high / (high + low))

########## Testing Results ##########
# electronics: 279801 high 33300 low 89% high
# movies and tv: 1289602 high 206629 low 86% high (usable?)
# CDs: 903002 high 92766 low 91% high
# clothing: 221597 high 26655 low 89% high
# home and kitchen: 455204 high 51419 low 89% high
# kindle store: 829277 high 57148 low 93% high
# sports and outdoors: 253017 high 19249 low 93% high
# cell phones: 148657 high 24343 low 86% high
# health: 279801 high 33300 low 89% high
# toys and games: 140235 high 11005 low 93% high
# video games: 174989 high 28516 low 86% high
# tools and home: 113602 high 10105 low 92% high
# beuty: 154272 21982 0.8752822630975751
# apps for android: 544718 high 123098 low 82% high (usable?)
# office products: 45342 high 2856 low 94% high
# pet supplies: 124248 high 17655 low 88% high
# grocery: 120044 high 13696 low 90% high
# baby: 126525 high 17012 low 88% high
# digital music: 52116 high 5801 low 90% high
# amazon instant video: 29336 high 3603 low 89% high



