import torch
from PIL import Image
import pandas as pd
import numpy as np
import util
import webmercator
import os
import sys


class DataInput(object):
    def __init__(self, test_data=False):
        self.test_data = test_data
        if self.test_data:
            self.image_dir = 'data/test_imagery/'
        else:
            self.image_dir = 'data/imagery/'

        self.x = None
        self.Y = None

    def load_data(self):
        print("loading data...")

        zip_codes = self.get_zip_codes()
        # print(zip_codes.head())
        # print(zip_codes.columns)

        tile_to_zip = {}

        for index, row in zip_codes.iterrows():
            # get the rounded x and y for the current zip code, which will match up with
            # the tile that contains the center of that zip code
            x, y = webmercator.xy(row['lat'], row['lon'], 14)
            x = round(x)
            y = round(y)

            tile_to_zip[(x, y)] = row['zip']

        zip_to_avg_income = {}
        for index, row in zip_codes.iterrows():
            # the Y value for each tile will be the average income for the zip code it
            # is a part of. Here, we just precompute the average incomes for each zip code
            zip_to_avg_income[row['zip']] = row['total_income'] / row['num_returns']

        data = []
        # get the list of file names in the specified directory
        files = os.listdir(self.image_dir)
        i = 0
        for filename in files:
            # increment counter for progress bar
            i += 1
            curr_entry = []

            # remove .jpg from end
            img_id = filename[:-4]
            x, y = util.get_coordinates(img_id)

            if (x, y) not in tile_to_zip:
                # this code is skipped for the tiles that have already been assigned a zip code
                # (because they contain the center of a zip code)
                lat, lon = webmercator.latlon(x, y, 14)

                if util.get_elevation(lat, lon) > 0:
                    # not an ocean tile
                    closest_zip = 0
                    min_dist = 987654321  # large number
                    for index, row in zip_codes.iterrows():
                        squared_dist = (lat - row['lat']) ** 2 + (lon - row['lon']) ** 2
                        if squared_dist < min_dist:
                            min_dist = squared_dist
                            closest_zip = row['zip']

                    tile_to_zip[(x, y)] = closest_zip

            if (x, y) in tile_to_zip:
                img = util.jpg_to_nparray(self.image_dir + filename)
                curr_entry.append(img)

                avg_income = zip_to_avg_income[tile_to_zip[(x, y)]]
                curr_entry.append(avg_income)

                data.append(curr_entry)

            # update progress bar
            percent = 100 * i / len(files)
            sys.stdout.write('\r')
            sys.stdout.write("Completed: [{:{}}] {:>3}%"
                             .format('=' * int(percent / (100.0 / 30)),
                                     30, int(percent)))
            sys.stdout.flush()

        # encapsulate data into a DataFrame
        data_frame = pd.DataFrame(data, columns=['image', 'avg_income'])

        self.x = data_frame['image']
        self.Y = data_frame['avg_income']

        print()
        print("loaded data")

        # print(self.x.head())
        # print(self.Y.head())

    def view_image(self, img_id):
        filename = self.image_dir + img_id + ".jpg"
        image = Image.open(filename).convert("RGB")
        image.show()

    def get_zip_codes(self):
        # load data
        tax_returns = pd.read_csv('data/16zpallnoagi.csv')
        zips = pd.read_csv('data/ziplatlon.csv', sep=';')

        # rename columns and select only the zip code, latitude, and longitude
        zips.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
        zip_locations = zips[['zip', 'lat', 'lon']]

        # rename columns and select only the zip code, number of tax returns, and total income
        tax_returns.rename(columns={'ZIPCODE': 'zip', 'N1': 'num_returns', 'A02650': 'total_income'}, inplace=True)
        income_by_zip = tax_returns[['zip', 'num_returns', 'total_income']]

        # merge the two DataFrames on the zip column
        zips = zip_locations.merge(income_by_zip, on='zip', how='left')
        # print("before restricting location: len(zips)=", len(zips))

        # restrict the zip codes to the right area
        zips = zips[round(webmercator.x(zips['lon'], 14)) >= 2794]
        zips = zips[round(webmercator.x(zips['lon'], 14)) <= 2839]
        zips = zips[round(webmercator.y(zips['lat'], 14)) >= 6528]
        zips = zips[round(webmercator.y(zips['lat'], 14)) <= 6572]
        zips = zips[zips['num_returns'] >= 0]
        zips = zips[zips['total_income'] >= 0]
        # print("after restricting location: len(zips)=", len(zips))
        return zips
