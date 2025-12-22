"""
# 分子3D可视化模块
"""
import os
import glob
import py3Dmol
from IPython.display import display, clear_output
import ipywidgets as widgets
from ipywidgets import HBox, VBox

def read_xyz_file(filename):
    """读取xyz文件的内容"""
    with open(filename, 'r') as f:
        content = f.read()
    return content

def view_molecule(xyz_file, view_container):
    """使用py3Dmol库可视化分子结构，支持动态旋转"""
    # 读取XYZ文件内容
    xyz_content = read_xyz_file(xyz_file)
    
    # 创建可视化窗口，设置大小
    view = py3Dmol.view(width=800, height=500)
    
    # 添加分子结构
    view.addModel(xyz_content, 'xyz')
    
    # 设置样式 - 球棒模型
    view.setStyle({'stick': {'radius': 0.2, 'color': 'grey'},
                   'sphere': {'scale': 0.3}})
    
    # 添加标签显示文件名
    file_name = os.path.basename(xyz_file)
    view.addLabel(file_name, {'position': {'x': 0, 'y': 0, 'z': 0}, 
                             'backgroundColor': 'white', 
                             'fontColor': 'black',
                             'backgroundOpacity': 0.5,
                             'fontSize': 18,
                             'alignment': 'bottomRight'})
    
    # 自动缩放视图以适应分子大小
    view.zoomTo()
    
    # 清除视图容器并显示新的视图
    with view_container:
        clear_output(wait=True)
        display(view)
        print(f"正在显示: {file_name} ({current_index+1}/{len(xyz_files)})")
        print("提示: 可以使用鼠标拖动旋转分子，滚轮缩放")
        print("      按「下一个」按钮或按Enter键显示下一个分子")

def see_xyz_interactive():
    """交互式XYZ文件查看器"""
    global current_index, xyz_files
    
    # 获取当前目录下所有的xyz文件
    current_dir = os.getcwd()
    xyz_files = sorted(glob.glob(os.path.join(current_dir, "*.xyz")))

    if not xyz_files:
        print("当前目录下未找到.xyz文件")
        return

    # 初始化当前索引
    current_index = 0
    
    # 创建下拉菜单以选择分子
    file_options = [(f"{i+1}. {os.path.basename(f)}", i) for i, f in enumerate(xyz_files)]
    file_dropdown = widgets.Dropdown(
        options=file_options,
        value=0,
        description='选择分子:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    # 创建按钮
    prev_button = widgets.Button(
        description='上一个',
        button_style='info',
        icon='arrow-left',
        layout=widgets.Layout(width='100px')
    )
    
    next_button = widgets.Button(
        description='下一个',
        button_style='info',
        icon='arrow-right',
        layout=widgets.Layout(width='100px')
    )
    
    # 创建状态显示
    status_label = widgets.Label(
        value=f'文件总数: {len(xyz_files)}'
    )
    
    # 创建输出区域
    view_container = widgets.Output()
    
    # 定义按钮回调函数
    def show_prev(b):
        global current_index
        if current_index > 0:
            current_index -= 1
            file_dropdown.value = current_index
            view_molecule(xyz_files[current_index], view_container)
    
    def show_next(b):
        global current_index
        if current_index < len(xyz_files) - 1:
            current_index += 1
            file_dropdown.value = current_index
            view_molecule(xyz_files[current_index], view_container)
    
    # 定义下拉菜单回调函数
    def on_dropdown_change(change):
        global current_index
        if change['type'] == 'change' and change['name'] == 'value':
            current_index = change['new']
            view_molecule(xyz_files[current_index], view_container)
    
    # 注册回调
    prev_button.on_click(show_prev)
    next_button.on_click(show_next)
    file_dropdown.observe(on_dropdown_change, names='value')

    # 创建控制面板
    controls = HBox([prev_button, next_button, file_dropdown, status_label])
    
    # 显示界面
    display(VBox([controls, view_container]))
    
    # 显示第一个分子
    view_molecule(xyz_files[current_index], view_container)