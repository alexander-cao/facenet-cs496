import os
import glob
import random


class TripletGenerator(object):
    anchor = 'anchor'
    positive = 'positive'
    negative = 'negative'

    def __init__(self, pubfig83_train_path = 'recognizer/pubfig83-train'):
        self.all_people = self.generate_all_people_dict(pubfig83_train_path)

    def generate_all_people_dict(self, pubfig83_train_path):
        # generates a dictionary between a person and all the photos of that person
        all_people = {}
        for person_folder in os.listdir(pubfig83_train_path):
            person_photos = glob.glob(pubfig83_train_path + os.path.sep + person_folder + os.path.sep + '*.jpg')
            all_people[person_folder] = person_photos
        # del all_people[".DS_Store"]
        return all_people

    def get_next_triplet(self):
        all_people_names = list(self.all_people.keys())

        while True:
            # draw a person at random for anchor
            anchor_person = random.choice(all_people_names)

            # repeatedly pick random person until we find one different from anchor
            negative_person = anchor_person
            while negative_person == anchor_person:
                negative_person = random.choice(all_people_names)

            anchor_photo = random.choice(self.all_people[anchor_person])
            # repeatedly pick random photo of anchor person until we find a different photo
            positive_photo = anchor_photo
            while positive_photo == anchor_photo:
                positive_photo = random.choice(self.all_people[anchor_person])

            negative_photo = random.choice(self.all_people[negative_person])

            yield ({self.anchor: anchor_photo, self.positive: positive_photo, self.negative: negative_photo})
