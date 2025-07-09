import numpy as np
from collections import Counter
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib

import torch

from options import _parse_args, more_about_config
from modules import build_model

def load_model(config_file,path,device):
    args, args_text = _parse_args(config_file)
    args.model_path = path

    args,_= more_about_config(args)

    model,_ = build_model(args, device, None)

    model_ckp =args.model_path
    model_ckp = torch.load(model_ckp,weights_only=True,map_location='cpu')

    if 'model' in model_ckp:
        _model_ckp = model_ckp['model']
    new_state_dict = {}
    for key in _model_ckp:
        # 将 'classifier.0.weight' 改为 'classifier.weight'
        if 'classifier.0.' in key:
            new_key = key.replace('classifier.0.', 'classifier.')
        elif '_fc2.' in key:
            new_key = key.replace('_fc2.','classifier.')
        else:
            new_key = key
        new_state_dict[new_key] = _model_ckp[key]
    _model_ckp = new_state_dict
    if not args.distributed:
        new_state_dict = {}
        for _k,v in _model_ckp.items():
            _k = _k.replace('module.','') if 'module' in _k else _k
            new_state_dict[_k]=v
    else:
        new_state_dict = _model_ckp

    info = model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()
    return model

def get_label_new(coords, mask_data, center_type=0):
    label = []

    for coord in coords:
        # 截取当前坐标对应的256x256的区域
        crop_region = mask_data.crop((coord[0], coord[1], coord[0] + 256, coord[1] + 256))

        # 将图像转换为numpy数组以便更高效地处理
        crop_array = np.array(crop_region)

        # 只统计第一个通道的值
        pixel_values = crop_array[:, :, 0].flatten()

        # 使用Counter统计每个像素值出现的次数
        pixel_counter = Counter(pixel_values)

        # 根据center_type选择不同的标签选择逻辑
        if center_type == 0:
            # 检查3, 4, 5是否存在
            high_priority_values = {3, 4, 5}
            high_priority_pixels = {k: v for k, v in pixel_counter.items() if k in high_priority_values}

            if high_priority_pixels:
                # 如果3, 4, 5中有值，选择其中最多的
                most_common_high = max(high_priority_pixels.items(), key=lambda x: x[1])
                label.append(most_common_high[0])
            else:
                # 否则选择0, 1, 2中最多的
                label.append(pixel_counter.most_common(1)[0][0])

        elif center_type == 1:
            # 检查是否存在值2
            if 2 in pixel_counter:
                label.append(2)
            else:
                # 否则选择最常见的值
                label.append(pixel_counter.most_common(1)[0][0])

    return np.array(label)

def plot(
        x,
        y,
        ax=None,
        title=None,
        draw_legend=True,
        draw_centers=False,
        draw_cluster_labels=False,
        colors=None,
        legend_kwargs=None,
        label_order=None,
        **kwargs
):
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    # plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}
    plot_params = {"alpha": kwargs.get("alpha", 0.8)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    # fixed_colors = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red'}
    default_colors = plt.cm.tab10.colors
    fixed_colors = {
        0: default_colors[0],  # blue
        1: default_colors[1],  # orange
        2: default_colors[2],  # green
        3: default_colors[3],  # red
        4: default_colors[4],  # purple
        5: default_colors[5],  # brown
    }
    if colors is None:
        max_label = max(classes)
        if max_label <= 2:
            colors = {0: '#FFEA7F', 1: '#4682b4', 2: '#ff7f7f'}
        else:
            blue_shades = ['#add8e6', '#87ceeb', '#4682b4']	 # light blue to darker blue
            red_shades = ['#ffb6c1', '#ff7f7f', '#ff0000']	 # light red to darkred
            colors = {i: blue_shades[i] if i <= 2 else red_shades[i-3] for i in classes}

    point_colors = list(map(colors.get, y))

    size = deepcopy(y)
    point_size = np.full_like(size, 50)
    for label in classes:
        if np.sum(y == label) / len(y) < 0.1:
            point_size[y == label] = 120

    fig = ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, s=point_size, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label + ': ' + str(len(x)),
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=str(yi) + ': ' + str(len(y[y == yi])),
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    return fig