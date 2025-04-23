from matplotlib import pyplot as plt

PARAMS = {}


def set_params(step, L1, L2, figsize):
    PARAMS['step'] = step
    PARAMS['L1'] = L1
    PARAMS['L2'] = L2
    PARAMS['figsize'] = figsize


def imshow(fig, ax, img, l1, l2, xlabel, ylabel, title):
    im = ax.imshow(img, extent=(-l2 / 2, l2 / 2, l1 / 2, -l1 / 2), aspect='auto', cmap='rainbow')
    ax.set_xlabel(ylabel)
    ax.set_ylabel(xlabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)


def vector_plot(fig, ax, x1, x2, v1, v2, title=None, xlabel='$x_i$', ylabel='$x_j$'):
    v_abs = (v1 ** 2 + v2 ** 2) ** 0.5
    col = ax.pcolor(x2, x1, v_abs, cmap='rainbow')
    fig.colorbar(col, ax=ax)

    ax.quiver(x2[::PARAMS['step'], ::PARAMS['step']],
              x1[::PARAMS['step'], ::PARAMS['step']],
              v2[::PARAMS['step'], ::PARAMS['step']],
              v1[::PARAMS['step'], ::PARAMS['step']])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)


def flow_visualization(v1, v2, x1, x2):
    fig, ax = plt.subplots(figsize=(PARAMS['figsize'], PARAMS['figsize']))

    vector_plot(fig, ax, x1,
                x2,
                v1, v2, xlabel='$x_2$', ylabel='$x_1$')
    plt.show()


def distribution_visualization(y, label):
    fig, ax = plt.subplots(figsize=(PARAMS['figsize'], PARAMS['figsize']))

    imshow(fig, ax,
           y,
           PARAMS['L1'],
           PARAMS['L2'],
           r'$x_1$',
           r'$x_2$',
           label)
    plt.show()


def train_history_plot(history):
    """Plot train history.

    Args:
        history (dict): Dict of lists with train history.
    """

    for i in history:
        fig, ax = plt.subplots(figsize=(PARAMS['figsize'] * 2, PARAMS['figsize']))

        ax.plot(history[i], c='r')
        ax.set_title(i)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('lr' if i == 'lr' else 'Loss')
        ax.legend(['Train'])
        if min(history[i]) > 0:
            ax.set_yscale('log')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
        plt.show()