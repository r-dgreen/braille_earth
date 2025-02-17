from argparse import ArgumentParser
from collections import defaultdict
import math
import os
from pathlib import Path

import numpy as np
from pyproj.transformer import Transformer
import shapefile
import tqdm


NE_10M_COASTLINE = Path(__file__).parent / 'ne_10m_coastline.zip'
NE_110M_ADMIN_0_COUNTRIES = Path(__file__).parent / 'ne_110m_admin_0_countries.shp'

BRAILLE_GRID = (
    (0/8, 1/17, 3/8, 4/17),
    (0/8, 5/17, 3/8, 8/17),
    (0/8, 9/17, 3/8, 12/17), 
    (4/8, 1/17, 8/8, 4/17),
    (4/8, 5/17, 8/8, 8/17),
    (4/8, 9/17, 8/8, 12/17),
    (0/8, 13/17, 3/8, 16/17),
    (4/8, 13/17, 8/8, 16/17))
CELL_RATIO = 8/17
POINT_ROUND = 2/8, 2/17

B0: int = 0x2800
B1: int = 0x0001
B2: int = 0x0002
B3: int = 0x0004
B4: int = 0x0008
B5: int = 0x0010
B6: int = 0x0020
B7: int = 0x0040
B8: int = 0x0080


def calculate_bg_bbox(width, height, x_offset, y_offset):
    for x1, y1, x2, y2 in BRAILLE_GRID:
        bbox = (x_offset + (x1 * width), 
                y_offset - (y1 * height), 
                x_offset + (x2 * width), 
                y_offset - (y2 * height))
        yield bbox      


def round_to_multiple(number, multiple):
    number = round(number) + (multiple - 1)
    return number - (number % multiple)


class BrailleEarth:
    def __init__(self, *shapefile, bbox=None):
        self.shapefiles = shapefile
        self.tranformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
        self.reverse_transformer = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
        if bbox is None:
            self.bbox = self.WEST, self.SOUTH, self.EAST, self.NORTH
        else:
            self.bbox = self.tranformer.transform_bounds(*bbox)
        self.set_resolution()
        

    def set_resolution(self, w=None, h=None):
        if w is None and h is None:
            w, _ = os.get_terminal_size()

        if w:
            self.columns = int(w)
            self.cell_width = (self.east - self.west) / self.columns
            self.cell_height = self.cell_width / CELL_RATIO
            self.rows = int((self.north - self.south) // self.cell_height)
        elif h:
            self.rows = int(h)
            self.cell_height = (self.north - self.south) / self.rows
            self.cell_width = self.cell_height * CELL_RATIO
            self.columns = int((self.east - self.west) // self.cell_width)

        self.bbox_array = np.zeros((self.columns, self.rows, 8, 4))
        self.braile_array = np.zeros((self.columns, self.rows, 8), dtype=np.bool_)

        x_origin = self.west
        y_origin = self.north
        for row in range(self.rows):
            y_offset = y_origin - (row * self.cell_height)
            for column in range(self.columns):
                x_offset = x_origin + (column * self.cell_width) 
                for index, bbox in enumerate(calculate_bg_bbox(width=self.cell_width, height=self.cell_height, x_offset=x_offset, y_offset=y_offset)):
                    self.bbox_array[column, row, index] = bbox
        
        x_region_size = 25
        y_region_size = 15
        self.x_regions = []
        self.y_regions = []
        for i in range(math.ceil(self.columns / x_region_size)):
            self.x_regions.append((i * x_region_size, (i * x_region_size) + x_region_size))
        for i in range(math.ceil(self.rows / y_region_size)):
            self.y_regions.append((i * y_region_size, (i * y_region_size) + y_region_size))

    def load_points(self):
        # round every point to the nearest cursor size multiple
        round_width, round_height = POINT_ROUND
        round_width *= self.cell_width
        round_height *= self.cell_height
        round_width, round_height = round(round_width), round(round_height)
        x_offset = (self.cell_width * self.columns) - self.east
        y_offset = (self.cell_height * self.rows) + self.south
        
        self.points = defaultdict(set)
        
        for sf_path in self.shapefiles:
            sf = shapefile.Reader(sf_path)
            for (left, right) in self.x_regions:
                right = min(right, self.columns)
                for (top, bottom) in self.y_regions:    
                    bottom = min(bottom, self.rows)
                    bbox = self.reverse_transformer.transform_bounds((left * self.cell_width) - x_offset,
                                                                    -((bottom * self.cell_height) - y_offset),
                                                                    (right * self.cell_width) - x_offset,
                                                                    -((top * self.cell_height) - y_offset))
                    for shape in sf.shapes(bbox=bbox):
                        for x, y in shape.points:
                            x, y = self.tranformer.transform(x, y)
                            x = round_to_multiple(x, round_width)
                            y = round_to_multiple(y, round_height)
                            self.points[x, y].add(((left, right), (top, bottom)))                        

        total = sum([len(value) for value in self.points.values()])

        progress_bar = tqdm.tqdm(total=total, ascii='⡀⡄⡆⡇⣇⣧⣷⣿')

        for (x, y), slices in self.points.items():
            for column_slice, row_slice in slices:
                column_slice = slice(*column_slice)
                row_slice = slice(*row_slice)

                progress_bar.update(n=1)

                # if any point falls within the bounding box then it counts as a clip
                self.braile_array[column_slice,row_slice,:] |= (x >= self.bbox_array[column_slice,row_slice,:,0]) & \
                                                            (x < self.bbox_array[column_slice,row_slice,:,2])  & \
                                                            (y <= self.bbox_array[column_slice,row_slice,:,1]) & \
                                                            (y > self.bbox_array[column_slice,row_slice,:,3])
        self.generate_output()

    def generate_output(self):
        self.output_array = self.braile_array[:,:,0] * B1 | \
                            self.braile_array[:,:,1] * B2 | \
                            self.braile_array[:,:,2] * B3 | \
                            self.braile_array[:,:,3] * B4 | \
                            self.braile_array[:,:,4] * B5 | \
                            self.braile_array[:,:,5] * B6 | \
                            self.braile_array[:,:,6] * B7 | \
                            self.braile_array[:,:,7] * B8
        self.output_array += B0

    def print(self):
        for row in self.output_array.T:
            for character in row:
                print(chr(character), end='')
            print()

    @property
    def NORTH(self):
        _, n = self.tranformer.transform(0, 85.06)
        return n

    @property
    def EAST(self):
        e, _ = self.tranformer.transform(180, 0)
        return e

    @property
    def SOUTH(self):
        _, s = self.tranformer.transform(0, -85.06)
        return s

    @property
    def WEST(self):
        w, _ = self.tranformer.transform(-180, 0)
        return w
    
    @property
    def north(self):
        return self.bbox[3]
    
    @property
    def east(self):
        return self.bbox[2]
    
    @property
    def south(self):
        return self.bbox[1]
    
    @property
    def west(self):
        return self.bbox[0]


def main():
    parser = ArgumentParser(description='Generate Braille Map from ShapeFiles. [EPSG:4326] -> [EPSG:3857]')
    parser.add_argument('shapefiles', metavar='SHAPEFILES', nargs='+', help='Paths to supported shapefile/shapefile archive.')
    parser.add_argument('-b', '--bbox', metavar=('LEFT', 'BOTTOM', 'RIGHT', 'TOP'), nargs=4, type=float, default=(-180, -60, 180, 75), help='GPS bounding box to clip region')
    args = parser.parse_args()
    be = BrailleEarth(*args.shapefiles, bbox=args.bbox)
    be.load_points()
    be.print()


if __name__ == '__main__':
    main()
