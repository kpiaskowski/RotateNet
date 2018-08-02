import os



def read_all_filenames(path):
    filenames = os.listdir(path)
    unique_names= set([name.split('.')[0] for name in filenames])
    data = {}
    for name in unique_names:
        if name + '.mtl' in filenames and name + '.obj' in filenames:
            data[name] = (os.path.join(path, name, '.obj'), os.path.join(path, name, '.obj'))
    return data

def read_classes(path):
    classes = sorted(os.listdir(path))
    return classes

def filenames_for_class(class_path):
    with open(class_path) as f:
        names = sorted([line.split(',')[0].lstrip('wss.') for line in f.readlines()[1:]])
        return names

objs_path = '/home/carlo/Downloads/models'
textures_path = '/home/carlo/Downloads/textures'
class_dir = './class_files'


data_dict = read_all_filenames(objs_path)
print(os.listdir(textures_path))
print(data_dict.keys())
# classes = read_classes(class_dir)
#
# data_path = './data' # -> generate data here, function input
# for cls in classes:
#     names = filenames_for_class(class_dir + '/' + cls)
#     for name in names:
#         if not name in data_dict.keys():
#             continue
#         # os.mkdir(data_path + '/' + cls + '_' + str(i))
#         obj_file = data_dict[name][0] # 0 for obj, 1 for mtl
#         print(obj_file)




        # kazdy obiekt niech bedzie w innym folderze
        # nazwy folderow
        # klasa_numerporzadkowy
        # nazwy plikow
        # image_index_rotx_roty_rotz_png