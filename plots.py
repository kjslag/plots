import collections
import numbers
import collections.abc as abc
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt

import mpl_toolkits.axes_grid1

# import matplotlib.pyplot as plt
# plt.plot([0,1,2], [2,4,6], label='line')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('title')
# plt.yscale('log')
# plt.ylim(1, 7)
# plt.legend()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# plt.imshow([[-1, 0, 1, 2]], cmap='gray', vmin=0, vmax=1)
# plt.colorbar()
# plt.show()
# plt.imshow(np.array([[[0.5,1,1], [1,1,1], [0,0,0]]], dtype=np.float32))
# plt.colorbar()
# plt.axis('off')
# plt.show()

class Line:
    def __init__(self, _x_or_y=None, /, y=None, label=None, *, x=None, fit=False, args=None, **kwargs):
        if _x_or_y is not None:
            if y is None:
                y = _x_or_y
            else:
                assert x is None
                x = _x_or_y
        assert y is not None

        self.x = tensor_items(x)
        self.y = tensor_items(y)
        self.label = label
        self.fit = fit
        self.args = args
        self.kwargs = kwargs

# example:
# line = Line((2, 6), [3,20,100], 'exp', fit=True)
# plot([
#     [1,2,3],
#     ([2,4,2], 'line1'),
#     ([2,3,5], [2,5,2], 'line2'),
#     Line([3,1,3], label='label', args=['-'], color='black'),
#     line,
# ], 'log', args=['-o'], data_range=(-1,4), labels=('x', 'y', 'title'))
# line.fit
def plot(lines, 
         scale='linear', *,
         labels=(None,None,None),
         data_range=None, # (x_min, x_max)
         subplot=plt, figsize=(9,5),
         plot_range=None, # ([x_min, x_max], [y_min, y_max])
         new_figure=True,
         legend=True, # or a dict of legend options
         args=['o-'],
         **kwargs):

    if len(lines) == 0:
        print(f'plots.plot: nothing to plot: {lines}')
        return
    
    if not isinstance(lines, list) or isinstance(lines[0], numbers.Number):
        lines = [lines]

    use_legend = False
    if scale == 'log':
        scale = ('linear', 'log')
    elif isinstance(scale, str):
        scale = (scale, scale)
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if not isinstance(line, (tuple, Line)):
            lines[i] = Line(x=data_range, y=line)
        elif isinstance(line, tuple):
            assert len(line) >= 1
            if len(line) == 1 or not (hasattr(line[1], '__getitem__') and isinstance(line[1][0], (numbers.Number, torch.Tensor))):
                lines[i] = Line(data_range, *line)
            else:
                lines[i] = Line(*line)
        line = lines[i]

        if line.x is None:
            line.x = np.arange(1, len(line.y)+1)
        elif isinstance(line.x, tuple): # x = (min, max, optional: # points)
            if len(line.x) == 2:
                line.x = (*line.x, len(line.y))
            line.x = np.linspace(*line.x)

        if line.fit is True or isinstance(line.fit, dict):
            fit_kwargs = line.fit if isinstance(line.fit, dict) else {}
            scale_fn = dict(linear=lambda x: x, log=np.log)
            unscale_fn = dict(linear=lambda x: x, log=np.exp)
            x = scale_fn[scale[0]](line.x)
            y = scale_fn[scale[1]](line.y)
            line.fit = scipy.stats.linregress(x, y)
            x = np.array([x[0], x[-1]])
            y = unscale_fn[scale[1]](line.fit.slope * x + line.fit.intercept)
            if 'label' not in fit_kwargs:
                fit_kwargs['label'] = f'{line.label} fit'
            if 'args' not in fit_kwargs:
                fit_kwargs['args'] = ['-']
            lines.insert(i+1,
                Line([line.x[0], line.x[-1]], y, **fit_kwargs)
            )
            i += 1

        i += 1

    is_plt = subplot is plt
    if new_figure and is_plt:
        plt.figure(figsize=figsize)
    for line in lines:
        use_legend |= line.label is not None

        if isinstance(line.x, str):
            plotter = subplot.bar
            subplot.xticks(rotation=90)
        else:
            plotter = subplot.plot

        line_args = line.args if line.args is not None else args

        assert len(line.x) == len(line.y)
        try:
            plotter(line.x, line.y, *line_args, label=line.label, **(kwargs | line.kwargs))
        except:
            raise ValueError(f'Error plotting {line}')

    if scale[0] != 'linear':
        (plt.xscale if is_plt else subplot.set_xscale)(scale[0])
    (plt.yscale if is_plt else subplot.set_yscale)(scale[1])
    
    if use_legend and legend is not False:
        subplot.legend(**legend) if isinstance(legend, dict) else subplot.legend()

    assert isinstance(labels, abc.Iterable)
    for fn, l in zip((plt.xlabel if is_plt else subplot.set_xlabel,
                      plt.ylabel if is_plt else subplot.set_ylabel,
                      plt.title  if is_plt else subplot.set_title), labels):
        if l is not None:
            fn(l)

    if plot_range is not None:
        if plot_range[0] is not None:
            plt.xlim(*plot_range[0]) if is_plt else subplot.set_xlim(*plot_range[0])
        if plot_range[1] is not None:
            plt.ylim(*plot_range[1]) if is_plt else subplot.set_ylim(*plot_range[1])

    if new_figure and is_plt:
        plt.show()

    return subplot

# example:
# plots.plot_data(list(range(8)), lambda x: x, lambda x: x**2, lambda x: x%2,
#     labels=['x', 'y', 'title'])
def plot_data(dataset, x_fn, y_fn, label_fn, *args,
              legend_labels=[], # in order to specify ordering
              options_fn=lambda label: {},
              post_process=lambda line: line,
              subplot=plot,
              **kwargs):
    xyls_dict = collections.defaultdict(list)
    def sortable_label(label):
        try:
            return (legend_labels.index(label), str(l))
        except ValueError:
            (len(legend_labels), str(l))
    for d in dataset:
        l = label_fn(d)
        if l is not None:
            xyls_dict[(sortable_label(l), l)].append((x_fn(d), y_fn(d)))
    for k in xyls_dict:
        xyls_dict[k].sort()
    plot_data = [Line([x for x,_ in xys], [y for _,y in xys], str(l), **options_fn(l))
                 for (_,l), xys in sorted(xyls_dict.items())]
    return subplot(post_process(plot_data), *args, **kwargs)

def gridplot_data(dataset, x_fn, y_fn, label_fns, *args, grid_kwargs={}, **kwargs):
    if not isinstance(label_fns[0], list):
        label_fns = [label_fns]
    
    subplots = [ [subplot_data(dataset, x_fn, y_fn, f, *args, **kwargs) for f in fs] for fs in label_fns ]
    grid(*subplots, **grid_kwargs)

def subplot_data(*args, **kwargs):
    return plot_data(*args, subplot=SubPlot, **kwargs)

# example:
# plots.grid([
#     plots.SubPlot([1,2,3]),
#     plots.SubPlot([2,3,4]),
# ], figsize=6)
def grid(*subplots, figsize=16):
    subplots = np.matrix(subplots) # NOTE: use grid([...], [...]) instead of grid([[...], [...]])
    n_rows, n_cols = subplots.shape
    if isinstance(figsize, int):
        figsize = ( figsize, figsize * n_rows/n_cols * 0.8 * n_cols/(1+n_cols) )
    fig, axs = plt.subplots(*subplots.shape, figsize=figsize)
#     fig, axs = plt.subplots(*subplots.shape, figsize=(figsize, figsize * (1+n_rows)/(1+n_cols)))
    axs = np.matrix(axs).reshape(*subplots.shape)

    for i in range(n_rows):
        for j in range(n_cols):
            sub = subplots[i,j]
            sub.plot(*sub.args, **sub.kwargs, subplot=axs[i,j])
    
    plt.tight_layout(pad=1*n_rows/n_cols)
#     plt.tight_layout(pad=10*n_rows/n_cols)
    plt.show()

class Sub:
    def __init__(self, plot, *args, **kwargs):
        self.plot   = plot
        self.args   = args
        self.kwargs = kwargs

class SubPlot(Sub):
    def __init__(self, *args, **kwargs):
        super().__init__(plot, *args, **kwargs)

class SubImage(Sub):
    def __init__(self, *args, **kwargs):
        super().__init__(image, *args, **kwargs)

def images(images, grid_figsize=16, **kwargs):
    grid(*[SubImage(image, **kwargs) for image in images], figsize=grid_figsize)

def image(image, title=None, subplot=plt, figsize=(9,5), legend=None, **kwargs):
    C, H, W = image.shape

    is_plt = subplot is plt
    if is_plt:
        plt.figure(figsize=figsize)

    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu().numpy()

    if C == 1:
        image = image[0]
        if 'cmap' not in kwargs:
            kwargs = dict(cmap='gray', vmin=0, vmax=1) | kwargs
        elif legend is None:
            legend = True
    elif C == 3:
        image = image.transpose(1, 2, 0)
    else:
        assert False

    im = subplot.imshow(image, **kwargs)
    if legend:
        if is_plt:
            plt.colorbar()
        else:
            cax = mpl_toolkits.axes_grid1.make_axes_locatable(subplot).append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
    if title is not None:
        subplot.set_title(title)
    subplot.axis('off')
    if is_plt:
        plt.show()

def hist(x, *args, show=True, **kwargs):
    kwargs = kwargs | dict(bins='auto', density=True, histtype='step')
    plt.hist(tensor_items(x), *args, **kwargs)
    if show:
        plt.show()

def tensor_items(xs):
    """Recursively convert tensors contained in xs to numbers or numpy arrays."""
    if isinstance(xs, list):
        return [tensor_items(x) for x in xs]
    elif isinstance(xs, tuple):
        return tuple(tensor_items(x) for x in xs)
    elif isinstance(xs, dict):
        return {k: tensor_items(v) for k,v in xs.items()}
    elif isinstance(xs, torch.Tensor):
        dtype = xs.dtype
        if dtype == torch.bfloat16:
            dtype = torch.float32
        return xs.item() if xs.dim()==0 else xs.detach().to(dtype=dtype, device='cpu').numpy()
    elif isinstance(xs, np.ndarray):
        return xs
    elif hasattr(xs, '__next__') and not hasattr(xs, '__getitem__'):
        return (tensor_items(x) for x in xs)
    else:
        return xs
