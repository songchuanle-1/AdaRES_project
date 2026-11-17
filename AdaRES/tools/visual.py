import numpy as np
import json
import open3d as o3d
def rle_decode(rle_dict):
    """从 RLE 字典解码回二进制掩码"""
    length = rle_dict['length']
    counts = list(map(int, rle_dict['counts'].split()))
    mask = np.zeros(length, dtype=np.uint8)
    current_pos = 0
    for i in range(0, len(counts), 2):
        start = counts[i] - 1  # RLE 编码从1开始计数
        end = start + counts[i+1]
        mask[start:end] = 1
        current_pos = end
    return mask

file_name = '/bak_home/songcl/3D_visual_grounding/IPDN/checkpoint_3/model_no_task/20250418_034023/results/pred_instance/predicted_masks/scene0011_00_[1]_3.json'
with open('file_name', 'r') as f:
    file_json = json.load(f)
    decoded_mask  = rle_decode(file_json)

pcdfile = '/bak_home/songcl/3D_visual_grounding/data/data/scannetv2/val/scene0011_00_vh_clean_2.ply'
# 读取点云
pcd = o3d.io.read_point_cloud(pcdfile)

# 创建Visualizer对象
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)  # 无头模式（需系统支持OpenGL离屏渲染）
vis.add_geometry(pcd)

# 渲染并保存
vis.update_geometry(pcd)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("output.jpg")  # 直接保存为JPG
vis.destroy_window()