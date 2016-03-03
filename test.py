import os
import nibabel
import numpy
import matplotlib
import mayavi
from mayavi import mlab
from mayavi.core import lut_manager


# get geo_path
# --------------------------------------------------------------------------------
subject_dir = "/nfs/j3/userhome/chenxiayu/workingdir/S0001"
hemi = "lh"
surf = "inflated"
geo_path = os.path.join(subject_dir, "surf", "%s.%s" % (hemi, surf))


# 读取坐标和面片信息coords, faces
# --------------------------------------------------------------------------------
# coords（numpy.ndarray：154592*3，float64）每一行为一个顶点的坐标，共有154592个顶点
# faces （numpy.ndarray：309180*3，int32）所有元素值的范围都在0和154591的闭区间内
# faces 每一个元素就是一个顶点号，对应coords的行号，每一行表示这三个顶点组成一个三角面片
coords, faces = nibabel.freesurfer.read_geometry(geo_path)


# 计算顶点法向量vtx_normals
# --------------------------------------------------------------------------------
# vtx1_s是vtx1的复数（英文中的单复数），表示这个变量包含多个顶点坐标
vtx1_s = coords[faces[:, 0], :]  # 意思是取出faces第一列所有顶点号对应的坐标
vtx2_s = coords[faces[:, 1], :]  # 道理同上
vtx3_s = coords[faces[:, 2], :]  # 道理同上
# 综上所述，vtx1_s、vtx2_s、vtx3_s的规格都是（numpy.ndarray：309180*3，float64）
# 可以知道vtx1_s、vtx2_s、vtx3_s相同行号的三个坐标表示的分别是组成三角面片的三个顶点

vtx1_to_vtx2_s = vtx2_s - vtx1_s  # 对于每一行来说，就相当于是每一个面片的vtx2-vtx1产生向量vtx1_to_vtx2
vtx1_to_vtx3_s = vtx3_s - vtx1_s  # 道理同上

# 对vtx1_to_vtx2_s和vtx1_to_vtx3_s相同行号的向量做叉乘得到对应面片的垂直向量,定义为落在顶点vtx1_s上
if max([vtx1_to_vtx2_s.shape[0], vtx1_to_vtx3_s.shape[0]]) >= 500:
    tri_normals = numpy.c_[vtx1_to_vtx2_s[:, 1] * vtx1_to_vtx3_s[:, 2] - vtx1_to_vtx2_s[:, 2] * vtx1_to_vtx3_s[:, 1],
                        vtx1_to_vtx2_s[:, 2] * vtx1_to_vtx3_s[:, 0] - vtx1_to_vtx2_s[:, 0] * vtx1_to_vtx3_s[:, 2],
                        vtx1_to_vtx2_s[:, 0] * vtx1_to_vtx3_s[:, 1] - vtx1_to_vtx2_s[:, 1] * vtx1_to_vtx3_s[:, 0]]
else:
    tri_normals = numpy.cross(vtx1_to_vtx2_s, vtx1_to_vtx3_s)

sizes = numpy.sqrt(numpy.sum(tri_normals * tri_normals, axis=1))  # 求各垂直向量的模
zero_idx = numpy.where(sizes == 0)[0]  # 找到sizes中值为0的索引数组
# 叉乘结果等于零表明叉乘的两个向量是平行的（这在三角面片中应该是不存在的，对于本次数据确实不存在，因此zero_idx是空的）
sizes[zero_idx] = 1.0  # 防止0值用作除数
tri_normals /= sizes[:, numpy.newaxis]  # numpy.newaxiss是为数组添加一个新的维度，这里使得sizes由行变成列
#上式的计算结果是得到各个垂直向量的单位向量，即各个面片的法向量

nvtxs = coords.shape[0]  # 顶点数量
vtx_normals = numpy.zeros((nvtxs, 3))

# 在各个顶点上将落在该顶点上的面法向量的x，y，z分量分别相加，实际效果是向量的合并，成为属于该顶点的向量
for vtxs in faces.T:
    for idx in range(3):
        vtx_normals[:, idx] += numpy.bincount(vtxs, tri_normals[:, idx], minlength=nvtxs)
# 单位化
sizes = numpy.sqrt(numpy.sum(vtx_normals * vtx_normals, axis=1))
sizes[sizes == 0] = 1.0
vtx_normals /= sizes[:, numpy.newaxis]


# deal with making figures
# --------------------------------------------------------------------------------
# set window properties
background = 'g'
foreground = 'w'
size = 800
try:
    width, height = size
except(TypeError, ValueError):
    width, height = size, size

bg_color = matplotlib.colors.colorConverter.to_rgb(background)
fg_color = matplotlib.colors.colorConverter.to_rgb(foreground)

# make viewer
title = "%s.%s" % (hemi, surf)
figure = mlab.figure(title, size=(width, height))  # 新建一个场景，会立马显示一个带有工具栏的窗口
mayavi.mlab.clf(figure)  # 不消除figure对象，也不关闭窗口，难道只是是figure保持“干净”？

if figure.scene is not None:
    figure.scene.background = bg_color
    figure.scene.foreground = fg_color

figure.render()  # force rendering so scene.lights exists，将之前的改动渲染到figure对象中，保证光照效果
mayavi.mlab.draw(figure=figure)  # 重绘figure

# toggle_toolbars
show_toolbar = False
if figure.scene is not None:
    if hasattr(figure.scene, "scene_editor"):
        bar = figure.scene.scene_editor._tool_bar
    else:
        bar = figure.scene._tool_bar

    if hasattr(bar, "setVisible"):
        bar.setVisible(show_toolbar)
    elif hasattr(bar, "Show"):
        bar.Show(show_toolbar)

figure.render()  # force rendering so scene.lights exists
mayavi.mlab.draw(figure=figure)  # 重绘figure

# toggle render
state = False
if mayavi.mlab.options.backend != 'test':
    figure.scene.disable_render = not state  # 这是在做什么？


# get curv_path
# --------------------------------------------------------------------------------
curv_path = os.path.join(subject_dir, "surf", "%s.curv" % hemi)


# read curvature information
# --------------------------------------------------------------------------------
curv = nibabel.freesurfer.read_morph_data(curv_path)
bin_curv = numpy.array(curv > 0, numpy.int)


# fill figures with brains
# --------------------------------------------------------------------------------
cortex = "classic"

# get colors
cortex_map = {
    "classic": ("Greys", -1, 2, False),
    "high_contrast": ("Greys", -.1, 1.3, False),
    "low_contrast": ("Greys", -5, 5, False),
    "bone": ("bone", -.2, 2, True)
}
# 在这里解释一下为什么"low_contrast"对应的是("Greys", -5, 5, False)，其它的举一反三都改知道是什么含义了
# 首先colormap是颜色映射的意思，也就是将颜色和scalar value做映射。"Grey"指定了颜色映射表用的是灰色，
# -5~5指定了映射表映射的值的范围，这里是将灰色映射到-5~5宽度为10的值域，而二值化的曲率的值只是0和1，它们之间
# 的灰度差只占了整体的1/10，所以对比度较低，然而high_contrast的值域是-.1~1.3，宽度为1.4，可想而知！
if cortex in cortex_map.keys():
    color_data = cortex_map[cortex]
elif cortex in lut_manager.lut_mode_list():
    color_data = cortex, -1, 2, False
else:
    color_data = cortex

# collect key-word args
colormap, vmin, vmax, reverse = color_data
meshargs = dict(figure=figure, scalars=bin_curv)  # 第二个参数是取了0,1两个标量
surfargs = dict(colormap=colormap, vmin=vmin, vmax=vmax)
# 获取坐标信息
x = coords[:, 0]
y = coords[:, 1]
z = coords[:, 2]

geo_mesh = mayavi.mlab.pipeline.triangular_mesh_source(x, y, z, faces, **meshargs)  # 返回一个二维网格
# 再根据顶点法向量调整成surface？
# add surface normals
geo_mesh.data.point_data.normals = vtx_normals
geo_mesh.data.cell_data.normals = None  # cell_data是指什么？

geo_surf = mayavi.mlab.pipeline.surface(geo_mesh, figure=figure, reset_zoom=True, **surfargs)

if reverse:
    curv_bar = mlab.scalarbar(geo_surf)  # get the scalar color map from geo_surf
    curv_bar.reverse_lut = True  # reverse the lut
    curv_bar.visible = False

for idx in range(faces.shape[0]):
    row = faces[idx, :]
    if 1 in row:
        print(row)

input()
