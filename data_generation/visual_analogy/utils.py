import numpy as np
import cv2
import copy
from numpy import random


CANVAS_SIZE = 160

DOMAINS = ['line_type', 'line_color', 'shape_type', 'shape_color', 'shape_size', 'shape_quantity', 'shape_position']

VALUES = {
    'line_type': {'falling diagonal line': (np.array([[0,0],[1,1]])*CANVAS_SIZE, 'line'), 
                  'rising diagonal line': (np.array([[0,1],[1,0]])*CANVAS_SIZE, 'line'), 
                  'horizontal line': (np.array([[0,0.5],[1,0.5]])*CANVAS_SIZE, 'line'), 
                  'vertical line': (np.array([[0.5,0],[0.5,1]])*CANVAS_SIZE, 'line'), 
                  'diamond lines': (np.array([[0.5,0],[0,0.5],[0.5,1],[1,0.5]])*CANVAS_SIZE, 'contour'),
                  'circular line': (np.array([[0.5,0.5]])*CANVAS_SIZE, 'circle'), 
                  'V-shape facing up': (np.array([[0,0],[0.5,1],[1,0]])*CANVAS_SIZE, 'contour'),
                  'V-shape facing left': (np.array([[0,0],[1,0.5],[0,1]])*CANVAS_SIZE, 'contour'),
                  'V-shape facing down': (np.array([[0,1],[0.5,0],[1,1]])*CANVAS_SIZE, 'contour'),
                  'V-shape facing right': (np.array([[1,0],[0,0.5],[1,1]])*CANVAS_SIZE, 'contour')
                  },
    'line_color': [0, 90, 135, 189],
    'shape_type': ['circle', 'rectangle', 'triangle', 'pentagon', 'hexagon'], 

    'shape_color': [0, 90, 135, 189, 255],
    'shape_size': np.arange(20, 42, 7), 
    'shape_quantity': np.arange(5)+1, 
    'shape_position': {'left of the top row': np.array([0.2,0.2])*CANVAS_SIZE, 
                       'middle of the top row': np.array([0.5,0.2])*CANVAS_SIZE, 
                       'right of the top row': np.array([0.8,0.2])*CANVAS_SIZE,
                       'left of the middle row': np.array([0.2,0.5])*CANVAS_SIZE, 
                       'center': np.array([0.5,0.5])*CANVAS_SIZE,
                       'right of the middle row': np.array([0.8,0.5])*CANVAS_SIZE,
                       'left of the bottom row': np.array([0.2,0.8])*CANVAS_SIZE, 
                       'middle of the bottom row': np.array([0.5,0.8])*CANVAS_SIZE, 
                       'right of the bottom row': np.array([0.8,0.8])*CANVAS_SIZE}
}

RELATIONS = ['XOR', 'OR', 'AND', 'Progression']

positions = ['left of the top row', 'middle of the top row', 'right of the top row', 
             'left of the middle row', 'center', 'right of the middle row', 
             'left of the bottom row', 'middle of the bottom row', 'right of the bottom row']
position_coordinates = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
pos2coord = dict(zip(positions, position_coordinates))
coord2pos = dict(zip(position_coordinates, positions))
label2color = {'black' : 0,
        'dark grey': 90,
        'grey': 135,
        'light grey': 189,
        'white': 255}
color2label = dict(list(map(lambda x: (x[1], x[0]), list(label2color.items()))))

# Hyperparameters for controlling task complexity
NUM_LINES = 2
NUM_SHAPES = 3


class Line:
    def __init__(self, meta_label=None):
        self.object_type = 'line'
        self.exist = True
        try:
            available_types = meta_label['available type'] if meta_label and 'available type' in meta_label else list(VALUES['line_type'].keys())
            forbid_types = meta_label['forbid type'] if meta_label and 'forbid type' in meta_label else []
            self.get_type(available_types, forbid_types)
            available_colors = meta_label['available color'] if meta_label and 'available color' in meta_label else VALUES['line_color']
            forbid_colors = meta_label['forbid color'] if meta_label and 'forbid color' in meta_label else []
            self.get_color(available_colors, forbid_colors)
        except:
            self.no_feasible_line()

    def get_type(self, available_types, forbid_types):
        self.type = np.random.choice(list(set(available_types).difference(forbid_types)))
        self.pts, self.mode = VALUES['line_type'][self.type]

    def get_color(self, available_colors, forbid_colors):  
        if forbid_colors:
            available_colors = list(set(available_colors).difference(forbid_colors))
        self.color = np.random.choice(available_colors)
        self.color_label = color2label[self.color]

    def get_info(self):
        return {'type': self.type, 'color_label': self.color_label, 'color': self.color}

    def get_info_tuple(self):
        return (self.type, self.color_label, self.color)
    
    def no_feasible_line(self):
        self.exist = False
    
    def get_attribute(self, attribute):
        if attribute == 'type':
            return self.type
        elif attribute == 'color':
            return self.color
        else:
            return None
    

class Shape:
    def __init__(self, meta_label=None):
        self.object_type = 'shape'
        self.exist = True
        try:
            available_types = meta_label['available type'] if meta_label and 'available type' in meta_label else list(VALUES['shape_type'])
            forbid_types = meta_label['forbid type'] if meta_label and 'forbid type' in meta_label else []
            self.get_type(available_types, forbid_types)
            available_colors = meta_label['available color'] if meta_label and 'available color' in meta_label else VALUES['shape_color']
            forbid_colors = meta_label['forbid color'] if meta_label and 'forbid color' in meta_label else []
            self.get_color(available_colors, forbid_colors)
            available_sizes = meta_label['available size'] if meta_label and 'available size' in meta_label else list(VALUES['shape_size'])
            forbid_sizes = meta_label['forbid size'] if meta_label and 'forbid size' in meta_label else []
            self.get_size(available_sizes, forbid_sizes)
            available_positions = meta_label['available position'] if meta_label and 'available position' in meta_label else position_coordinates
            forbid_positions = meta_label['forbid position'] if meta_label and 'forbid position' in meta_label else []
            self.get_position(available_positions, forbid_positions)
        except:
            self.no_feasible_shape()
    
    def get_type(self, available_types, forbid_types):
        self.type = np.random.choice(list(set(available_types).difference(forbid_types)))

    def get_color(self, available_colors, forbid_colors):   
        if forbid_colors:
            available_colors = list(set(available_colors).difference(forbid_colors))
        self.color = np.random.choice(available_colors)
        self.color_label = color2label[self.color]

    def get_size(self, available_sizes, forbid_sizes):
        self.size = np.random.choice(list(set(available_sizes).difference(forbid_sizes)))

    def get_position(self, available_positions, forbid_positions):
        positions = list(set(available_positions).difference(forbid_positions))
        self.position_coords = positions[np.random.choice(len(positions))]
        self.position_label = coord2pos[self.position_coords]
        self.position = VALUES['shape_position'][self.position_label]

    def get_info(self):
        return {'type': self.type, 'color_label': self.color_label, 'color': self.color, 'size': self.size, 'position_coordinate': self.position_coords, 'position_label': self.position_label}
    
    def get_info_tuple(self):
        return (self.type, self.color_label, self.color, self.size, self.position_coords, self.position_label)
    
    def no_feasible_shape(self):
        self.exist = False

    def get_attribute(self, attribute):
        if attribute == 'type':
            return self.type
        elif attribute == 'color':
            return self.color
        elif attribute == 'size':
            return self.size
        elif attribute == 'position':   
            return self.position_coords
        else:
            return None


class Panel:
    def __init__(self, canvas_size, progression_direction=None):
        self.canvas_size = canvas_size
        self.objects = []
        self.img = None
        self.shape_quantity = 0
        self.forbid_shape_type = np.random.choice(VALUES['shape_type'])
        self.forbid_shape_color = np.random.choice(VALUES['shape_color'])
        self.forbid_shape_size = np.random.choice(VALUES['shape_size'])
        self.forbid_shape_position = position_coordinates[np.random.choice(len(position_coordinates))]
        self.progression_direction = progression_direction  # optional argument specifying the direction of progression if following progression pattern

    def add_object(self, object):
        if object.exist:
            self.objects.append(object)
    
    def delete_objects(self, indexes):
        retain_indexes = [i for i in range(len(self.objects)) if i not in indexes]
        self.objects = [self.objects[i] for i in retain_indexes]
    
    def add_n_shapes(self, n, meta_label=None):
        # add n shapes without overlapping shapes
        if not meta_label:
            meta_label = {}
        new_shape_indexes = []
        if 'available type' not in meta_label and 'forbid type' not in meta_label:
            meta_label['forbid type'] = [self.forbid_shape_type]
        if 'available color' not in meta_label and 'forbid color' not in meta_label:
            meta_label['forbid color'] = [self.forbid_shape_color]
        if 'available size' not in meta_label and 'forbid size' not in meta_label:
            meta_label['forbid size'] = [self.forbid_shape_size]
        if 'available position' not in meta_label and 'forbid position' not in meta_label:
            meta_label['forbid position'] = [self.forbid_shape_position]

        for _ in range(n):
            forbid_positions = set(meta_label.get('forbid position', []))
            forbid_positions = forbid_positions.union(self.get_all_positions())
            meta_label['forbid position'] = forbid_positions
            new_shape = Shape(meta_label)
            if not new_shape.exist:
                self.delete_objects(new_shape_indexes)
                return False    # fail to add n shapes
            self.add_object(Shape(meta_label))
            new_shape_indexes.append(len(self.objects)-1)
        return True

    def get_all_types(self):
        return self.get_all_attributes('type')
    
    def get_all_colors(self):
        return self.get_all_attributes('color')
    
    def get_all_sizes(self):
        return self.get_all_attributes('size')
    
    def get_all_positions(self):
        return self.get_all_attributes('position')
    
    def get_all_quantities(self):
        assert(len([object for object in self.objects if type(object) == Shape]) == self.shape_quantity)
        return self.shape_quantity
    
    def get_all_objects_info(self):
        info = dict()
        objects = []
        for i, object in enumerate(self.objects):
            objects.append((i, object.object_type, object))
        for i, object_type, object in objects:
            info['{}{}'.format(object_type, i)] = object.get_info()
        if self.shape_quantity:
            info['quantity'] = self.shape_quantity
        return info

    def get_all_objects_info_tuple(self):
        objects = []
        for object in self.objects:
            objects.append(object.get_info_tuple())
        return objects
    
    def get_all_attributes(self, attribute):
        if attribute == 'quantity':
            return [self.get_all_quantities()]
        else:
            return [object.get_attribute(attribute) for object in self.objects]

    def draw(self):
        self.img = np.ones((self.canvas_size, self.canvas_size), np.uint8) * 255
        for object in self.objects:
            if type(object) == Line:
                self.img = draw_line(self.img, object.pts, object.color, object.mode)
            else:   # object is Shape
                if object.type == 'circle':
                    self.img = draw_circle(self.img, object.position, object.size, object.color)
                elif object.type == 'rectangle':
                    self.img = draw_rectangle(self.img, object.position, object.size, object.color)
                else:
                    self.img = draw_polygon(self.img, object.type, object.position, object.size, object.color)
        return self.img


def draw_line(img, pts, color, mode):
    pts = pts.astype(int)
    color = int(color)
    if mode == 'line':
        pt0, pt1 = pts
        img = cv2.line(img, pt0, pt1, color, 4)
    elif mode == 'circle':
        img = cv2.circle(img, pts[0], int(CANVAS_SIZE/5), color, 4)
    else:   # mode == 'contour'
        img = cv2.drawContours(img, [pts], -1, color, 4)
    return img

def draw_circle(img, center, size, color):
    size = int(size)
    color = int(color)
    x, y = center
    img = cv2.circle(img, (int(x),int(y)), int(size/2), color, cv2.FILLED)
    img = cv2.circle(img, (int(x),int(y)), int(size/2), 0, 2)
    return img

def draw_rectangle(img, center, size, color):
    x, y = center
    color = int(color)
    pt1 = (int(x-size/2), int(y-size/2))
    pt2 = (int(x+size/2), int(y+size/2))
    img = cv2.rectangle(img, pt1, pt2, color, cv2.FILLED)
    img = cv2.rectangle(img, pt1, pt2, 0, 2)
    return img

def draw_polygon(img, shape, center, size, color):
    color = int(color)
    if shape == 'semicircle':
        x, y = center
        img = cv2.ellipse(img, (int(x),int(y)), (int(size/2),int(size/2)), 0, 0, 180, color, cv2.FILLED)
        img = cv2.ellipse(img, (int(x),int(y)), (int(size/2),int(size/2)), 0, 0, 180, 0, 2)
        pt1 = (int(x-size/2), int(y))
        pt2 = (int(x+size/2), int(y))
        img = cv2.line(img, pt1, pt2, 0, 2)
        return img
    elif shape == 'triangle-up' or shape == 'triangle':
        coords = [[0,-0.5], [0.577,0.5], [-0.577,0.5]]
    elif shape == 'triangle-down':
        coords = [[0,0.5], [-0.577,-0.5], [0.577,-0.5]]
    elif shape == 'triangle-left':
        coords = [[-0.5,0], [0.5,0.577], [0.5,-0.577]]
    elif shape == 'triangle-right':
        coords = [[0.5,0], [-0.5,0.577], [-0.5,-0.577]]
    elif shape == 'pentagon':
        coords = [[0,-0.5], [-0.525,-0.12], [-0.324,0.496], [0.324,0.496], [0.525,-0.12]]
    elif shape == 'hexagon':
        coords = [[0.5,0], [0.25,-0.45], [-0.25,-0.45], [-0.5,0], [-0.25,0.45], [0.25,0.45]]
    else:
        coords = [[0.207,0.5], [0.5,0.207], [0.5,-0.207], [0.207,-0.5], [-0.207,-0.5], [-0.5,-0.207], [-0.5,0.207], [-0.207,0.5]]    
    img = cv2.drawContours(img, [(np.array(coords)*size+center).astype(int)], -1, color, cv2.FILLED)
    img = cv2.drawContours(img, [(np.array(coords)*size+center).astype(int)], -1, 0, 2)
    return img


def create_context(domain, canvas_size=CANVAS_SIZE):
    object_type, target_attribute = domain.split('_')
    panel1 = Panel(canvas_size)
    common_object = None
    if object_type == 'line':
        n_lines = np.random.randint(NUM_LINES)
        common_object = Line()      # panel 1 and 2 share the same attribute value for this object (intersection)
        panel1.add_object(common_object)
        diff_meta_label = {}
        # diff object has different attribute value from the common object
        diff_meta_label['forbid {}'.format(target_attribute)] = [common_object.get_attribute(target_attribute)]
        forbid_types = set(diff_meta_label.get('forbid type', []))
        forbid_types = forbid_types.union(panel1.get_all_types())
        diff_meta_label['forbid type'] = list(forbid_types)
        diff_object = Line(diff_meta_label)     # panel 1 and 2 have different attribute value for this object (difference)
        panel1.add_object(diff_object)
        for _ in range(n_lines):
            panel1.add_object(Line({'forbid type': panel1.get_all_types()}))
    else:   # object_type == 'shape'
        common_object = Shape()
        panel1.add_object(common_object)
        diff_meta_label = {}
        if target_attribute == 'color':
            diff_meta_label['forbid color'] = [common_object.color]
        else:
            diff_meta_label['forbid {}'.format(target_attribute)] = [common_object.get_attribute(target_attribute)]
        forbid_positions = set(diff_meta_label.get('forbid position', []))
        forbid_positions = forbid_positions.union(panel1.get_all_positions())
        diff_meta_label['forbid position'] = list(forbid_positions)
        diff_object = Shape(diff_meta_label)
        panel1.add_object(diff_object)
        panel1.shape_quantity = np.random.randint(2, NUM_SHAPES)
        panel1.add_n_shapes(panel1.shape_quantity-2)

    panel2 = Panel(canvas_size)
    if object_type == 'line':
        meta_label = {}
        meta_label['available {}'.format(target_attribute)] = [common_object.get_attribute(target_attribute)]
        meta_label['forbid {}'.format(target_attribute)] = [diff_object.get_attribute(target_attribute)]
        panel2.add_object(Line(meta_label))
        del meta_label['available {}'.format(target_attribute)]

        n_lines = np.random.randint(NUM_LINES)
        for _ in range(n_lines):
            forbid_type = set(meta_label.get('forbid type', []))
            forbid_type = forbid_type.union(panel2.get_all_types())
            meta_label['forbid type'] = list(forbid_type)
            panel2.add_object(Line(meta_label))
    else:   # object_type == 'shape'
        meta_label = {}
        meta_label['available {}'.format(target_attribute)] = [common_object.get_attribute(target_attribute)]
        meta_label['forbid {}'.format(target_attribute)] = [diff_object.get_attribute(target_attribute)]
        panel2.add_object(Shape(meta_label))
        del meta_label['available {}'.format(target_attribute)]

        panel2.shape_quantity = np.random.randint(2, NUM_SHAPES)
        panel2.add_n_shapes(panel2.shape_quantity-1, meta_label)

    return panel1, panel2


def create_progression_context(domain, canvas_size=CANVAS_SIZE, progression_direction=None):
    object_type, target_attribute = domain.split('_')

    # create panel 1
    panel1 = Panel(canvas_size, progression_direction)
    if object_type == 'line':
        if target_attribute == 'color':
            if not panel1.progression_direction:
                panel1.progression_direction = np.random.choice(['from dark color to light color', 'from light color to dark color'])
            panel1_colors = VALUES['line_color']
            if panel1.progression_direction == 'from dark color to light color':
                panel1_colors = panel1_colors[:-2]  # leave at least 3 colors for panel 2 and 3
            else: # from light color to dark color
                panel1_colors = panel1_colors[2:]
        
            common_line = Line({'available color': panel1_colors}) # at least 1 line type in common to avoid empty intersection
            panel1.add_object(common_line)
            panel1.add_object(Line({'available color': panel1.get_all_colors(), 'forbid type': panel1.get_all_types()}))
            if np.random.random() > 0.5:
                panel1.add_object(Line({'available color': panel1.get_all_colors(), 'forbid type': panel1.get_all_types()}))
        else:
            panel1.add_object(Line())
            if np.random.random() > 0.5:
                panel1.add_object(Line({'forbid type': panel1.get_all_types()}))
    else:  # object_type == 'shape'
        if target_attribute == 'type' or target_attribute == 'position':
            panel1.shape_quantity = np.random.choice(VALUES['shape_quantity'][:-2])     # leave at least two slots for new shape type / position
        elif target_attribute == 'quantity':
            if not panel1.progression_direction:
                panel1.progression_direction = np.random.choice(['increase in quantity', 'decrease in quantity'])
            if panel1.progression_direction == 'increase in quantity':
                panel1.shape_quantity = np.random.choice(VALUES['shape_quantity'][:-2]) # leave at least 2 spaces for panel 2 and 3
            else:  # decrease in quantity
                panel1.shape_quantity = np.random.choice(VALUES['shape_quantity'][2:])
        else:   # target_attribute == 'color' or 'size'
            panel1.shape_quantity = np.random.choice(VALUES['shape_quantity'])
        if target_attribute == 'color' or target_attribute == 'size':
            if not panel1.progression_direction:
                if target_attribute == 'color':
                    panel1.progression_direction = np.random.choice(['from dark color to light color', 'from light color to dark color'])
                else:  # 'size'
                    panel1.progression_direction = np.random.choice(['increase in size', 'decrease in size'])
            if panel1.progression_direction == 'from dark color to light color' or panel1.progression_direction == 'increase in size':
                attributes = VALUES['shape_{}'.format(target_attribute)][:-2]  # leave at least 3 attribute values for panel 2 and 3           
            else:  # from light color to dark color or decrease in size
                attributes = VALUES['shape_{}'.format(target_attribute)][2:]
            meta_label = {'available {}'.format(target_attribute): np.random.choice(attributes, 1)}
        else:
            meta_label = None
        panel1.add_n_shapes(panel1.shape_quantity, meta_label)

    # create panel 2
    panel2 = Panel(canvas_size, progression_direction)
    if object_type == 'line':
        if target_attribute == 'type':
            for object in panel1.objects:
                panel2.add_object(Line({'available type': [object.type]}))
            panel2.add_object(Line({'forbid type': panel1.get_all_types()}))    # new type
        else:  # target_attribute == 'color'
            if not panel2.progression_direction:
                direction = panel2.progression_direction = panel1.progression_direction
            panel2_colors = np.array(VALUES['line_color'])
            index = np.where(panel2_colors == panel1.get_all_colors()[0])[0][0]
            if direction == 'from dark color to light color':
                panel2_colors = panel2_colors[index+1:]
            else:  # from light color to dark color
                panel2_colors = panel2_colors[:index]
            # exclude the lightest and darkest colors
            panel2_colors = panel2_colors[np.where(panel2_colors != VALUES['line_color'][0])[0]]
            panel2_colors = panel2_colors[np.where(panel2_colors != VALUES['line_color'][-1])[0]]
            # color progression needs at least 1 line type in common to avoid empty intersection
            panel2.add_object(Line({'available color': np.random.choice(panel2_colors, 1), 'available type': [common_line.type]}))
            panel2.add_object(Line({'available color': panel2.get_all_colors(), 'forbid type': panel2.get_all_types()}))
            if np.random.random() > 0.5:
                panel2.add_object(Line({'available color': panel2.get_all_colors(), 'forbid type': panel2.get_all_types()}))
    else:  # object_type == 'shape'
        if target_attribute == 'quantity':
            if not panel2.progression_direction:
                direction = panel2.progression_direction = panel1.progression_direction
            panel2_quantities = VALUES['shape_quantity']
            index = np.where(VALUES['shape_quantity'] == panel1.shape_quantity)[0][0]
            if direction == 'increase in quantity':
                panel2_quantities = panel2_quantities[index+1:]
            else:  # decrease in quantity
                panel2_quantities = panel2_quantities[:index]
            # exclude the smallest and largest quantity
            panel2_quantities = panel2_quantities[np.where(panel2_quantities != VALUES['shape_quantity'][0])[0]]
            panel2_quantities = panel2_quantities[np.where(panel2_quantities != VALUES['shape_quantity'][-1])[0]]
            panel2.shape_quantity = np.random.choice(panel2_quantities)
            panel2.add_n_shapes(panel2.shape_quantity)
        else:
            panel1_attributes = list(set(panel1.get_all_attributes(target_attribute)))
            panel2.shape_quantity = np.random.randint(len(panel1_attributes), np.max(VALUES['shape_quantity']))
            if target_attribute == 'type':
                # add at least one shape of the same type
                for t in panel1_attributes:
                    panel2.add_n_shapes(1, {'available type': [t]})
                # add the rest shapes
                panel2.add_n_shapes(panel2.shape_quantity-len(panel1_attributes))
                #print("panel 2 shape quantity", panel2.shape_quantity, len(panel2.objects))
            elif target_attribute == 'color':
                if not panel2.progression_direction:
                    direction = panel2.progression_direction = panel1.progression_direction
                panel2_colors = np.array(VALUES['shape_color'])
                index = np.where(panel2_colors == panel1.get_all_colors()[0])[0][0]
                if direction == 'from dark color to light color':
                    panel2_colors = panel2_colors[index+1:]
                else: # from light color to dark color
                    panel2_colors = panel2_colors[:index]
                # exclude the lightest and darkest colors
                panel2_colors = panel2_colors[np.where(panel2_colors != VALUES['shape_color'][0])[0]]
                panel2_colors = panel2_colors[np.where(panel2_colors != VALUES['shape_color'][-1])[0]]
                panel2.add_n_shapes(panel2.shape_quantity, {'available color': np.random.choice(panel2_colors, 1)})
                #print('panel 2 shape color', panel2.get_all_colors())
            elif target_attribute == 'size':
                if not panel2.progression_direction:
                    direction = panel2.progression_direction = panel1.progression_direction
                panel2_sizes = np.array(VALUES['shape_size'])
                index = np.where(VALUES['shape_size'] == panel1.get_all_sizes()[0])[0][0]
                if direction == 'increase in size':
                    panel2_sizes = panel2_sizes[index+1:]
                else:  # decrease in size
                    panel2_sizes = panel2_sizes[:index]
                # exclude the smallest and largest size
                panel2_sizes = panel2_sizes[np.where(panel2_sizes != VALUES['shape_size'][0])[0]]
                panel2_sizes = panel2_sizes[np.where(panel2_sizes != VALUES['shape_size'][-1])[0]]
                panel2.add_n_shapes(panel2.shape_quantity, {'available size': np.random.choice(panel2_sizes, 1)})
            else:   # target_attribute == 'position'
                panel2.shape_quantity = panel1.shape_quantity   # only position progresses
                if not panel1.progression_direction:
                    direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
                    panel1.progression_direction = panel2.progression_direction = direction
                for object in panel1.objects:
                    i, j = object.position_coords
                    if direction == 'horizontal':
                        j = (j+1)%3
                    elif direction == 'vertical':
                        i = (i+1)%3
                    else:  # diagonal
                        i = (i+1)%3
                        j = (j+1)%3
                    panel2.add_n_shapes(1, {'available position': [(i,j)]})
    
    return panel1, panel2


def create_progression(domain, panel1, panel2):
    object_type, target_attribute = domain.split('_')
    if target_attribute == 'type':
        types1 = panel1.get_all_types()
        types2 = panel2.get_all_types()
        if not(len(set(types2).difference(types1)) > 0 and len(set(types1).difference(types2)) == 0):  # panel 2 type needs to be a superset of panel 1
            return ([], 'any')
        types = set(types1).union(types2)
        all_types = set(list(VALUES['line_type'].keys())) if object_type == 'line' else set(VALUES['shape_type'])
        new_types = set(list(all_types)).difference(types)
        types.add(np.random.choice(list(new_types))) 
        return (list(types), 'all')
    elif target_attribute == 'color':
        if len(set(panel1.get_all_colors())) != 1 or len(set(panel2.get_all_colors())) != 1:    # panel 1 and panel 2 must be uni-color to progress, used only in the exclusion case
            return ([], 'any')
        if object_type == 'line':
            all_colors = VALUES['line_color']
        else:   # shape
            all_colors = VALUES['shape_color']
        index = np.where(all_colors == panel2.get_all_colors()[0])[0][0]
        if panel1.progression_direction == 'from dark color to light color' or panel1.get_all_colors()[0] < panel2.get_all_colors()[0]:    # panel 1 is darker than panel 2
            return (all_colors[index+1:], 'any')
        elif panel1.progression_direction == 'from light color to dark color' or panel1.get_all_colors()[0] > panel2.get_all_colors()[0]:   # panel 1 is lighter than panel 2
            return (all_colors[:index], 'any')
        else:   # panel 1 and panel 2 have the same color
            return ([panel1.get_all_colors()[0]], 'any')
    elif target_attribute == 'size':
        if len(set(panel1.get_all_sizes())) != 1 or len(set(panel2.get_all_sizes())) != 1:    # panel 1 and panel 2 must be uni-size to progress
            return ([], 'any')
        index = np.where(VALUES['shape_size'] == panel2.get_all_sizes()[0])[0][0]
        if panel1.progression_direction == 'increase in size' or panel1.get_all_sizes()[0] < panel2.get_all_sizes()[0]:
            return (VALUES['shape_size'][index+1:], 'any')
        elif panel1.progression_direction == 'decrease in size' or panel1.get_all_sizes()[0] > panel2.get_all_sizes()[0]:
            return (VALUES['shape_size'][:index], 'any')
        else:
            return ([panel1.get_all_sizes()[0]], 'any')
    elif target_attribute == 'quantity':
        index = np.where(VALUES['shape_quantity'] == panel2.get_all_quantities())[0][0]
        if panel1.progression_direction == 'increase in quantity' or panel1.get_all_quantities() < panel2.get_all_quantities():
            return (VALUES['shape_quantity'][index+1:], 'any')
        elif panel1.progression_direction == 'decrease in quantity' or panel1.get_all_quantities() > panel2.get_all_quantities():
            return (VALUES['shape_quantity'][:index], 'any')
        else:
            # do not check the case where both panels have the same quantity, as it is allowed for position progression
            # only used in the exclusion case
            return ([], 'any')
    else:   # target_attribute == 'position'
        panel1_positions = panel1.get_all_positions()
        panel2_positions = panel2.get_all_positions()
        # check if progression is possible
        if panel1_positions[0][0] == panel2_positions[0][0] and (panel1_positions[0][1]+1)%3 == panel2_positions[0][1]:
            direction = 'horizontal'
        elif panel1_positions[0][1] == panel2_positions[0][1] and (panel1_positions[0][0]+1)%3 == panel2_positions[0][0]:
            direction = 'vertical'
        elif (panel1_positions[0][0]+1)%3 == panel2_positions[0][0] and (panel1_positions[0][1]+1)%3 == panel2_positions[0][1]:
            direction = 'diagonal'
        else:   # impossible to progress
            return ([], 'any')
        for i, j in panel1_positions:
            if direction == 'horizontal' and (i, (j+1)%3) not in panel2_positions:
                return ([], 'any')
            elif direction == 'vertical' and ((i+1)%3, j) not in panel2_positions:
                return ([], 'any')
            else:
                if ((i+1)%3, (j+1)%3) not in panel2_positions:
                    return ([], 'any')
        position = []
        for i, j in panel2_positions:
            if panel1.progression_direction == 'horizontal':
                position.append((i, (j+1)%3))
            elif panel1.progression_direction == 'vertical':
                position.append(((i+1)%3, j))
            else:   # diagonal
                position.append(((i+1)%3, (j+1)%3))
        return (position, 'all')


def create_or(domain, panel1, panel2):
    _, target_attribute = domain.split('_')
    if target_attribute == 'type':
        types = set(panel1.get_all_types())
        types = types.union(panel2.get_all_types())
        return list(types)
    elif target_attribute == 'color':
        colors = set(panel1.get_all_colors()) 
        colors = colors.union(panel2.get_all_colors())
        return list(colors)
    elif target_attribute == 'size':
        sizes = set(panel1.get_all_sizes())
        sizes = sizes.union(panel2.get_all_sizes())
        return list(sizes)
    else:   # target_attribute == 'position'
        positions = set(panel1.get_all_positions())
        positions = positions.union(panel2.get_all_positions())
        return list(positions)


def create_xor(domain, panel1, panel2):
    _, target_attribute = domain.split('_')
    if target_attribute == 'type':
        types1 = set(panel1.get_all_types())
        types2 = set(panel2.get_all_types())
        all_types = types1.union(types2)
        common_types = types1.intersection(types2)
        return list(all_types.difference(common_types))
    elif target_attribute == 'color':
        colors1 = set(panel1.get_all_colors())
        colors2 = set(panel2.get_all_colors())
        all_colors = colors1.union(colors2)
        common_colors = colors1.intersection(colors2)
        colors = all_colors.difference(common_colors)
        return list(colors)
    elif target_attribute == 'size':
        sizes1 = set(panel1.get_all_sizes())
        sizes2 = set(panel2.get_all_sizes())
        all_sizes = sizes1.union(sizes2)
        common_sizes = sizes1.intersection(sizes2)
        return list(all_sizes.difference(common_sizes))
    else:  # target_attribute == 'position'
        positions1 = set(panel1.get_all_positions())
        positions2 = set(panel2.get_all_positions())
        all_positions = positions1.union(positions2)
        common_positions = positions1.intersection(positions2)
        return list(all_positions.difference(common_positions))
        

def create_and(domain, panel1, panel2):
    _, target_attribute = domain.split('_')
    if target_attribute == 'type':
        types = set(panel1.get_all_types())
        types = types.intersection(panel2.get_all_types())
        return list(types)
    elif target_attribute == 'color':
        colors = set(panel1.get_all_colors())
        colors = colors.intersection(panel2.get_all_colors())
        return list(colors)
    elif target_attribute == 'size':
        sizes = set(panel1.get_all_sizes())
        sizes = sizes.intersection(panel2.get_all_sizes())
        return list(sizes)
    else:   # target_attribute == 'position'
        positions = set(panel1.get_all_positions())
        positions = positions.intersection(panel2.get_all_positions())
        return list(positions)
    

def exclude_spurious_correlation(panel1, panel2, target_domain, target_relation, available_constraints, label_type):
    meta_label = {}
    target_object_type, target_attribute = target_domain.split('_')
    target_domains = [domain for domain in DOMAINS if domain.startswith(target_object_type)]
    for domain in target_domains:
        _, attribute = domain.split('_')
        for relation in RELATIONS:
            if domain == target_domain and relation == target_relation:
                continue
            elif domain == 'shape_quantity' and relation != 'Progression':
                continue
            else:
                forbid_list = meta_label.get('forbid {}'.format(attribute), [])
                constraint_type = 'all'
                if relation == 'Progression':
                    constraint, constraint_type = create_progression(domain, panel1, panel2)
                elif relation == 'OR':
                    constraint= create_or(domain, panel1, panel2)
                elif relation == 'XOR':
                    constraint = create_xor(domain, panel1, panel2)
                else:   # relation == 'AND'
                    constraint = create_and(domain, panel1, panel2)

                if constraint_type == 'all':
                    if len(constraint) > 0:
                        forbid_list = set(forbid_list)
                        constraint_index = np.random.choice(len(constraint))
                        if attribute == target_attribute and label_type == 'all':
                            exclude_set = set(constraint)
                            include_set = set(available_constraints)
                            if exclude_set == include_set:  # impossible to exclude all spurious correlation
                                return None
                            elif len(include_set.difference(exclude_set)) > 0:
                                continue
                            else:
                                exclude_constraints = list(exclude_set.difference(include_set))
                                forbid_list.add(exclude_constraints[np.random.choice(len(exclude_constraints))])
                                continue
                        forbid_list.add(constraint[constraint_index])
                        forbid_list = list(forbid_list)
                else:   # constraint_type == 'any'
                    forbid_list = set(forbid_list)
                    forbid_list = forbid_list.union(constraint)
                    forbid_list = list(forbid_list)
                meta_label['forbid {}'.format(attribute)] = forbid_list
    return meta_label
    

def check_repeat_options(options, option_panel):
    option_panel_objects = option_panel.get_all_objects_info_tuple()
    if len(options) == 0:
        return False
    for _, option in options:
        option_objects = option.get_all_objects_info_tuple()
        if set(option_objects) == set(option_panel_objects):
            return True
    return False


def sample_new_option_for_any(options, option_panel, meta_label, target_attribute, constraints, object_type):
    meta_label['available {}'.format(target_attribute)] = np.random.choice(constraints, 1)
    if object_type == 'line':
        line = Line(meta_label)
        if line.exist:
            option_panel.add_object(line)
            meta_label['forbid type'] = option_panel.get_all_types()
        else:
            return None
    else:  # object_type == 'shape'
        if not option_panel.add_n_shapes(option_panel.shape_quantity, meta_label):
            return None
        
    n_sample = 1
    feasible_option = True
    while (check_repeat_options(options, option_panel) or (not feasible_option)) and n_sample < 10:
        option_panel.delete_objects([-1])
        meta_label['available {}'.format(target_attribute)] = np.random.choice(constraints, 1)
        if object_type == 'line':
            line = Line(meta_label)
            if line.exist:
                option_panel.add_object(line)
                meta_label['forbid type'] = option_panel.get_all_types()
            else:
                feasible_option = False
        else:  # object_type == 'shape'
            if not option_panel.add_n_shapes(option_panel.shape_quantity, meta_label):
                feasible_option = False
        n_sample += 1

    if not check_repeat_options(options, option_panel) and feasible_option:
        return option_panel
    else:
        return None


def sample_new_option_for_all(options, option_panel, meta_label, target_attribute, constraints, object_type):
    constraints = list(map(lambda x: [x], constraints))

    new_object_indexes = []
    for c in constraints:
        meta_label['available {}'.format(target_attribute)] = c
        if object_type == 'line':
            line = Line(meta_label)
            if line.exist:
                option_panel.add_object(line)
                meta_label['forbid type'] = option_panel.get_all_types()    # avoid adding line of the same type that will overlap
            else:
                return None
        else:  # object_type == 'shape'
            if not option_panel.add_n_shapes(1, meta_label):
                return None
        new_object_indexes.append(len(option_panel.objects)-1)

    for _ in range(len(option_panel.objects), option_panel.shape_quantity):
        if len(constraints) > 0:
            meta_label['available {}'.format(target_attribute)] = constraints[np.random.choice(len(constraints))]
        if object_type == 'line':
            line = Line(meta_label)
            if line.exist:
                option_panel.add_object(line)
                meta_label['forbid type'] = option_panel.get_all_types()
            else:
                return None
        else: # object_type == 'shape'
            if not option_panel.add_n_shapes(1, meta_label):
                return None
        new_object_indexes.append(len(option_panel.objects)-1)

    n_sample = 1
    feasible_option = True
    while (check_repeat_options(options, option_panel) or (not feasible_option)) and n_sample < 10:
        feasible_option = True
        option_panel.delete_objects(new_object_indexes)
        for c in constraints:
            meta_label['available {}'.format(target_attribute)] = c
            if object_type == 'line':
                line = Line(meta_label)
                if line.exist:
                    option_panel.add_object(line)
                    meta_label['forbid type'] = option_panel.get_all_types()
                else:
                    feasible_option = False
                    break
            else: # object_type == 'shape'
                if not option_panel.add_n_shapes(1, meta_label):
                    feasible_option = False
                    break
        if not feasible_option:
            continue
                
        for _ in range(len(option_panel.objects), option_panel.shape_quantity):
            if len(constraints) > 0:
                meta_label['available {}'.format(target_attribute)] = constraints[np.random.choice(len(constraints))]
            if object_type == 'line':
                line = Line(meta_label)
                if line.exist:
                    option_panel.add_object(line)
                    meta_label['forbid type'] = option_panel.get_all_types()
                else:
                    feasible_option = False
                    break
            else: # object_type == 'shape'
                if not option_panel.add_n_shapes(1, meta_label):
                    feasible_option = False
                    break
        n_sample += 1

    if not check_repeat_options(options, option_panel) and feasible_option:
        return option_panel
    else:
        return None
    

def create_panel(panel1, panel2, options, domain, relation):
    option_panel = Panel(CANVAS_SIZE)
    object_type, target_attribute = domain.split('_')
    label_type = 'all'
    if relation == 'Progression': 
        constraints, label_type = create_progression(domain, panel1, panel2)
    elif relation == 'OR':
        constraints = create_or(domain, panel1, panel2)
    elif relation == 'XOR':
        constraints = create_xor(domain, panel1, panel2)
    else:   # relation == 'AND'
        constraints = create_and(domain, panel1, panel2)
    
    # if available set is empty, the relation is impossible
    if len(constraints) == 0:
        return options

    meta_label = exclude_spurious_correlation(panel1, panel2, domain, relation, constraints, label_type)
    if not meta_label:      # if cannot exclude spurious correlation
        return options
    if object_type == 'shape':
        if domain == 'shape_quantity':
            quantities = set(constraints)
        elif domain == 'shape_position' and relation == 'Progression':
            quantities = set([panel1.shape_quantity, panel2.shape_quantity])
            if len(quantities) != 1:    # both panels must have the same shape quantity in position progression
                return options
        elif domain == 'shape_position':    # if constraints are defined for position, then include exactly number of shapes as the number of available positions in the constraints
            quantities = set([len(constraints)])
        else:
            quantities = set(np.arange(len(constraints), min(max(VALUES['shape_quantity']), len(VALUES['shape_position'])))+1)  # shape quantities upper bounded by max quantity and the number of positions
        positions = position_coordinates
        if 'forbid quantity' in meta_label:
            quantities = quantities.difference(meta_label['forbid quantity'])
        if 'available position' in meta_label:
            positions = meta_label['available position']
        if 'forbid position' in meta_label:
            positions = list(set(positions).difference(meta_label['forbid position']))
        quantities = np.array(list(quantities))
        quantities = quantities[np.where(quantities <= len(positions))[0]]
        if len(quantities) == 0:
            return options
        option_panel.shape_quantity = np.random.choice(list(quantities))
    if label_type == 'any':
        option_panel = sample_new_option_for_any(options, option_panel, meta_label, target_attribute, constraints, object_type)
        if option_panel:
            options.append([(domain, relation), option_panel])
    else:
        option_panel = sample_new_option_for_all(options, option_panel, meta_label, target_attribute, constraints, object_type)
        if option_panel:
            options.append([(domain, relation), option_panel])
    return options


def create_panels(target_domain, target_relation, create_option=False, canvas_size=CANVAS_SIZE, progression_direction=None, option_patterns=None):
    target_object_type = target_domain.split('_')[0]
    if target_relation == 'Progression':
        panel1, panel2 = create_progression_context(target_domain, canvas_size, progression_direction)
    else:
        panel1, panel2 = create_context(target_domain, canvas_size)
    panels = [panel1, panel2]
    
    option_domains = [domain for domain in DOMAINS if domain.startswith(target_object_type)]
    if option_patterns:
        option_patterns = [pattern for pattern in option_patterns if pattern[0].startswith(target_object_type) and pattern[1] != target_relation]
    # XOR, OR, AND are not defined on shape quantity
    options = create_panel(panel1, panel2, [], target_domain, target_relation)
    gt_index = 0
    if len(options) != 1:   # if cannot generate gt option, return empty options
        return panels, options, gt_index
    
    if create_option:
        relations = np.array(RELATIONS)[np.where(np.array(RELATIONS) != target_relation)[0]]
        domain_relation_pairs = option_patterns if option_patterns else [(domain, relation) for domain in option_domains for relation in relations]
        domain_relation_pairs = np.random.permutation(domain_relation_pairs)
        for (domain, relation) in domain_relation_pairs:
            if domain == 'shape_quantity' and relation != 'Progression':
                continue
            options = create_panel(panel1, panel2, options, domain, relation)          
            if len(options) == 4:
                break
        random.shuffle(options)
        find_match = False
        for i, ((domain, relation), _) in enumerate(options):
            if domain == target_domain and relation == target_relation:
                gt_index = i
                find_match = True
                break
        if not find_match:
            return panels, [], gt_index     # if cannot generate gt option, return empty options
    return panels, options, gt_index
    
    