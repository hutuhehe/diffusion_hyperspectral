def get_palette(category):
    if category == 'Augsburg':
        return Augsburg_palette
    elif category == 'Berlin_benchmark':
        return Berlin_benchmark_palette
    elif category == 'hyperspectral_64':
        return hyperspectral_64_palette
    elif category == 'Berlin_64':
      return Berlin_64_palette


def get_class_names(category):
    if category == 'Augsburg':
        return Augsburg_class
    elif category == 'Berlin_benchmark':
        return Berlin_benchmark_class
 
    elif category =='hyperspectral_64':
        return hyperspectral_64_class
    elif category =='Berlin_64':
      return Berlin_64_class
        

###############
# Class names #
###############




Berlin_benchmark_class = [
    #'background',
    'Forest',               # Associated with forested areas
    'Residential Area',     # Often used for residential zones
    'Industrial Area',      # Indicates indstrial areas
    'Low Plants',           # For areas with low vegetation or grasslands
    'Soil',                 # Represents bare soil or non-vegetated earth
    'Allotment',            # For allotments or cultivated land
    'Commercial Area',      # For commercial districts
    'Water'                 # Universally recognized for water bodies
]


Augsburg_class= [
   #"background",
    'Forest',               # Associated with forested areas
    'Residential Area',     # Often used for residential zones
    'Industrial Area',      # Indicates indstrial areas
    'Low Plants',           # For areas with low vegetation or grasslands
    'Allotment',            # For allotments or cultivated land
    'Commercial Area',      # For commercial districts
    'Water'                 # Universally recognized for water bodies
]



hyperspectral_64_class = [
    #'background',
    'Grass healthy',  # "#56b936"
    'Grass stressed',  # "#97eb47"
    'Grass synthetic',  # "#437c52"
    'Tree',  # "#437c52"
    'Soil',  # "#8c4f31"
    'Water',  # "#6aebeb"
    'Residual',  # "#0123e3"
    'Commercial',  # "#ccb6cf"
    'Road',  # "#d82b22"
    'Highway',  # "#76140d"
    'Railway',  # "#dc39e7"
    'Parking Lot 1',  # "#edeb4e"
    'Parking Lot 2',  # "#d29031"
    'Tennis',  # "#48217b"
    'Running'  # "#ef7851"
]


Berlin_64_class = [
    #'background',
    "Surface Water",
    "Street Network",
    "Urban Fabric",
    "Industrial, Commercial, and Transport",
    "Mine, Dump, and Construction Sites",
    "Artificial Vegetated Areas",
    "Arable Land",
    "Permanent Crops",
    "Pastures",
    "Forests",
    "Shrub",
    "Open Spaces with no Vegetation",
    "Inland Wetlands"
]


###########
# Palette #
###########


"""
Augsburg_palette = [


    #0, 0, 0,        # Black for background - Assuming you still need this
    34, 139, 34,     # Green for Forest
    255, 0, 0,       # Red for Residential Area
    255, 215, 0,     # Yellow for Industrial Area (hex #FFD700 is yellow)
    124, 252, 0,     # Light Green for Low Plants (#7CFC00 is Lawn Green)
    53, 94, 59,      # Darker green for Allotment (#355E3B is a dark green)
    0, 255, 255,     # Aqua for Commercial Area (usually aqua is light cyan or #00FFFF)
    0, 0, 255        # Blue for Water
]
"""

Augsburg_palette = [
    0, 0, 0,        # Black for background
    34, 139, 34,    # Green for Forest (#228B22)
    244, 164, 96,   # Sandy Brown for Residential Area (#F4A460)
    178, 34, 34,    # Firebrick for Industrial Area (#B22222)
    60, 179, 113,   # Medium Sea Green for Low Plants (#3CB371)
    218, 165, 32,   # Goldenrod for Allotment (#DAA520)
    147, 112, 219,  # Medium Purple for Commercial Area (#9370DB)
    0, 191, 255     # Deep Sky Blue for Water (#00BFFF)
]



Berlin_benchmark_palette = [
    0, 0, 0,           # Black for background
    34, 139, 34,       # Dark Green for Forest
    250, 128, 114,     # Salmon for Residential Area
    255, 69, 0,        # OrangeRed for Industrial Area
    144, 238, 144,     # Light Green for Low Plants
    139, 69, 19,       # Saddle Brown for Soil
    255, 215, 0,       # Yellow for Allotment
    186, 85, 211,      # Medium Orchid for Commercial Area
    0, 191, 255        # Deep Sky Blue for Water
]




"""
Berlin_benchmark_palette = [
    #0, 0, 0,        # Black for background - Assuming you still need this
    34, 139, 34,     # Green for Forest
    255, 0, 0,       # Red for Residential Area
    255, 215, 0,     # Yellow for Industrial Area (hex #FFD700 is yellow)
    124, 252, 0,     # Light Green for Low Plants (#7CFC00 is Lawn Green)
    165, 42, 42,     # Brown for Soil
    53, 94, 59,      # Darker green for Allotment (#355E3B is a dark green)
    0, 255, 255,     # Aqua for Commercial Area (usually aqua is light cyan or #00FFFF)
    0, 0, 255        # Blue for Water
]

"""


hyperspectral_64_palette = [
   # 0, 0, 0,  # for background
    86, 185, 54,   # Grass healthy 1
    151, 235, 71,  # Grass stressed 2
    67, 124, 82,   # Grass synthetic 3, Tree 4 (same color)
    67, 124, 82,   # Tree 4 (repeated color for clarity)
    140, 79, 49,   # Soil 5
    106, 235, 235, # Water 6
    1, 35, 227,    # Residual 7
    204, 182, 207, # Commercial 8
    216, 43, 34,   # Road 9
    118, 20, 13,   # Highway 10
    220, 57, 231,  # Railway 11
    237, 235, 78,  # Parking Lot 1 12
    210, 144, 49,  # Parking Lot 2 13
    72, 33, 123,   # Tennis 14
    239, 120, 81   # Running 15
]

Berlin_64_palette = [
#0, 0, 0, # Background
0, 255, 255, # Surface Water
255, 255, 255, # Street Network
255, 0, 0, # Urban Fabric
221, 160, 221, # Industrial, Commercial, and Transport
128, 0, 128, # Mine, Dump, and Construction Sites
255, 105, 180, # Artificial Vegetated Areas
255, 215, 0, # Arable Land
210, 180, 140, # Permanent Crops
128, 128, 0, # Pastures
0, 255, 0, # Forests
85, 107, 47, # Shrub
165, 42, 42, # Open Spaces with no Vegetation
106, 90, 205, # Inland Wetlands

]


