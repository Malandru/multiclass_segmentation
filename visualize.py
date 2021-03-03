import matplotlib.pyplot as plt


def prepare_figure(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure()
    for i, (name, image) in enumerate(images.items()):
        if image.shape[2] == 1:
            image = image.squeeze()
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)


def show_prepared_figures():
    plt.show()
