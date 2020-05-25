from gym_minigrid.register import register
from gym_minigrid.index_mapping import malmo_object_to_index, minigrid_index_mapping
from pathlib import Path

import numpy as np
from .numpymap import NumpyMap


RESOURCES_DIR = (Path(__file__).parent / './resources').resolve()


def fix_levers_on_same_level(same_level, above_level):
    """
    Input: 3D numpy array with malmo_object_to_index mapping

    Returns:
        3D numpy array where 3 channels represent 
        object index, color index, state index 
        for minigrid
    """
    lever_idx = malmo_object_to_index['lever']
    condition = above_level == lever_idx 
    minimap_array = np.where(condition, above_level, same_level) 
    return minimap_array


def fix_jump_locations(same_level, above_level, minigrid_index_mapping, jump_index=11):
    """
    Input: 3D numpy array with malmo_object_to_index mapping

    Returns:
        1. 3D numpy array where 3 channels represent 
        object index, color index, state index 
        for minigrid
        2. updated minigrid_index_mapping
    
    Notation for jump location 
            index = 11
            object = box
            color = grey # like walls
            toggletimes = 1

        NOTE: toggle to a box is a substitute for jump action
    """

    
    wall_idx = malmo_object_to_index['stained_hardened_clay']
    empty_idx = malmo_object_to_index['air']

    condition1 = same_level == wall_idx 
    condition2 = above_level == empty_idx

    minigrid_index_mapping['object_mapping'][jump_index] = 'box'
    minigrid_index_mapping['color_mapping'][jump_index] = 'white' #'grey0' # minigrid_index_mapping['color_mapping'][wall_idx]
    minigrid_index_mapping['toggletimes_mapping'][jump_index] = 1

    # Elementwise product of two bool arrays for AND
    minimap_array = np.where(condition1 * condition2, jump_index, same_level) # jump_index is broadcasted!

    return minimap_array, minigrid_index_mapping


def get_minimap_from_voxel(raw_map, minigrid_index_mapping, jump_index=11):
    """
    Aggregates minimap obtained from different same_level and above_map transforms
    Input: 3D numpy array with malmo_object_to_index mapping

    Returns:
        1. 3D numpy array where 3 channels represent 
        object index, color index, state index 
        for minigrid
        2. updated minigrid_index_mapping
    
    Functioning:
        * fixes jump locations
        * fixes levers in same level

        NOTE: toggle to a box is a substitute for jump action
    """
    same_level = raw_map[1]
    above_level = raw_map[2]
    
    minimap_array, modified_index_mapping = fix_jump_locations(
        same_level, above_level, minigrid_index_mapping, jump_index)
    minimap_array = fix_levers_on_same_level(minimap_array, above_level)

    return minimap_array, modified_index_mapping



class MinimapForMinecraft(NumpyMap):

    def __init__(self, raw_map_path=Path(RESOURCES_DIR, 'raw_map.npy'), agent_pos=(4, 25), agent_dir=0):
        raw_map = np.load(raw_map_path)
        minimap_array, modified_index_mapping = get_minimap_from_voxel(
            raw_map, minigrid_index_mapping, jump_index=11)
        super().__init__(
            modified_index_mapping, minimap_array, 
            agent_pos=agent_pos, agent_dir=agent_dir)



register(
    id='MiniGrid-MinimapForMinecraft-v0',
    entry_point='gym_minigrid.envs:MinimapForMinecraft'
)

