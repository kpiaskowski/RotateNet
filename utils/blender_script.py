import bpy
import os
from math import radians as rad
import time 

def initial_cleaning():
    obj_objects = bpy.data.objects['Cube']
    bpy.ops.object.delete()
    bpy.data.objects['Lamp'].select = True
    bpy.ops.object.delete()
    
def place_camera():
    obj = bpy.data.objects['Camera']
    obj.location.x = 0.0
    obj.location.y = -11.0
    obj.location.z = 0.0  
    obj.rotation_euler = (rad(90), rad(0), rad(0))
    
def set_lamp_energy(energy):
    for lamp in bpy.data.lamps[1:]:
        lamp.energy = energy    
    
def set_rendering():
    scene = bpy.data.scenes["Scene"]
    # set render resolution
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.color_mode ='RGB'
    
def set_world_color(color):
    scene = bpy.data.scenes["Scene"]
    scene.world.horizon_color = (color, color, color)
    
def create_lamps():
    scene = bpy.context.scene
    
    lamp_xy_range = 7.0
    lamp_z_range = 5.0
    positions = [[-lamp_xy_range, -lamp_xy_range, -lamp_z_range], [lamp_xy_range, -lamp_xy_range, -lamp_z_range], [-lamp_xy_range, lamp_xy_range, -lamp_z_range], [lamp_xy_range, lamp_xy_range, -lamp_z_range], [-lamp_xy_range, -lamp_xy_range, lamp_z_range], [lamp_xy_range, -lamp_xy_range, lamp_z_range], [-lamp_xy_range, lamp_xy_range, lamp_z_range], [lamp_xy_range, lamp_xy_range, lamp_z_range]]

    for pos in positions:
        lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')
        lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)
        scene.objects.link(lamp_object)
        lamp_object.location = pos
        lamp_object.select = True
        scene.objects.active = lamp_object
    
root_dir = '/home/carlo/rotatenet/shapenet_raw'
dst_dir = '/media/carlo/My Files/Praca/Etap3/shapenet'

x_low = -40
x_high = 45
x_step = 10
x_offset = 90

z_start = 0
z_end = 360
z_step = 3

max_instances = 300

classes = list(sorted(os.listdir(root_dir)))

initial_cleaning()
place_camera()
create_lamps()
set_rendering()

set_lamp_energy(1)

for c, cls in enumerate(classes):
    obj_dirs = sorted([os.path.join(root_dir, cls, name) for name in os.listdir(os.path.join(root_dir, cls))])
    for i, obj_dir in enumerate(obj_dirs):
        if i > max_instances:
            break
        # creating directories
        os.mkdir(os.path.join(dst_dir, cls + '_{}'.format(i)))

        # load object
        file_loc = os.path.join(obj_dir, 'model.obj')
        imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
        obj_objects = bpy.context.selected_objects
        
        for obj_object in obj_objects:
            # scale object
            scaling_factor = 6.5
            obj_object.scale = (scaling_factor, scaling_factor, scaling_factor)
        
        # rotating object
        for x in range(x_low, x_high, x_step):
            for z in range(z_start, z_end, z_step):
                for obj_object in obj_objects:
                    obj_object.rotation_euler = (rad(x+x_offset),0,rad(z))
            
                # render
                # color img
                set_world_color(0.55)
                set_lamp_energy(1)
                path = '{}/{}_{}/{}_{}_rgb'.format(dst_dir, cls, i, x, z)
                bpy.data.scenes['Scene'].render.filepath = path
                bpy.ops.render.render(write_still=True)
                
                # mask
                set_world_color(1)
                set_lamp_energy(0)
                path = '{}/{}_{}/{}_{}_mask'.format(dst_dir, cls, i, x, z)
                bpy.data.scenes['Scene'].render.filepath = path
                bpy.ops.render.render(write_still=True)
                
        # cleaning memory
        for obj_object in obj_objects:
            bpy.ops.object.delete() 
            del obj_object 
        
        print('Class {} ({} of {}), file {} of {}'.format(cls, c + 1, len(classes), i + 1, len(obj_dirs)))   
        
    
