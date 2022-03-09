from trimesh import Scene
import numpy as np

def trimesh_scene_to_b3dm(scene: Scene, turn_180deg_around_x = True) -> bytes:
        if turn_180deg_around_x:
            scene.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
        magic_value = b"b3dm"
        version = 1
        tile_byte_length = 0
        ft_json = b'{"BATCH_LENGTH":0}  '
        ft_json_byte_length = len(ft_json)
        ft_bin_byte_length = 0
        bt_json_byte_length = 0
        bt_bin_byte_length = 0
        bt_length = 0  # number of models in the batch
        gltf_data = np.frombuffer(trimesh.exchange.gltf.export_glb(scene), dtype=np.uint8)
        tile_byte_length = 28 + len(ft_json) + len(gltf_data)
        header_arr = np.frombuffer(magic_value, np.uint8)
        header_arr2 = np.array([version,
                                tile_byte_length,
                                ft_json_byte_length,
                                ft_bin_byte_length,
                                bt_json_byte_length,
                                bt_bin_byte_length], dtype=np.uint32)
        header_arr3 = np.frombuffer(ft_json, np.uint8)
        header = np.concatenate((header_arr, header_arr2.view(np.uint8), header_arr3))
        #with open("gltf_test.glb", 'wb') as file:
        #    file.write(gltf_data)
        result = bytes(np.concatenate((header, gltf_data)))
        return result