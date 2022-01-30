import tensorflow as tf
import matplotlib.pyplot as plt


def plot_input_frame(**kwargs):
    """Plot input frame

    Returns:
        plt.image.ImageAxis: matplotlib object
    """
    if 'ax' in list(kwargs.keys()):
        kwargs['ax'].set_title("Input frame")
        return kwargs['ax'].imshow(kwargs['frame'][0, -1, :, :, :], cmap='gray'), kwargs['frame'][0, -1, :, :, :]
    else:
        kwargs['plot'].set_array(kwargs['frame'][0, -1, :, :, :])
        return kwargs['plot'], kwargs['frame'][0, -1, :, :, :]


def plot_predicted_frame(**kwargs):
    """Plot predicted frame

    Returns:
        plt.image.ImageAxis: matplotlib object
    """
    if 'ax' in list(kwargs.keys()):
        kwargs['ax'].set_title("Predicted frame")
        return kwargs['ax'].imshow(kwargs['pred'][0, -1, :, :, :], cmap='gray'), kwargs['pred'][0, -1, :, :, :]
    else:
        kwargs['plot'].set_array(kwargs['pred'][0, -1, :, :, :])
        return kwargs['plot'], kwargs['pred'][0, -1, :, :, :]


metric_names = {
    'mse': 'Mean Squared Error (MSE)',
    'mean squared error': 'Mean Squared Error (MSE)',
    'psnr': 'Peak Signal Noise Ratio (PSNR)',
    'peak signal noise ratio': 'Peak Signal Noise Ratio (PSNR)',
    'ssim': 'Structural Similarity Index Measure (SSIM)'
}


def plot_anomaly_metric(**kwargs):
    """Plot anomaly metric, including its saved history

    Returns:
        plt.lines.Line2D: plot history
    """
    metric = kwargs['plot_name']
    if 'ax' in list(kwargs.keys()):
        kwargs['ax'].set_title(f'{metric_names[metric]} anomaly score')
        kwargs['ax'].set_xlabel('frame')
        kwargs['ax'].set_ylabel(metric)
        kwargs['ax'].set_xlim(0, kwargs['data_size'])
        kwargs['ax'].set_ylim(0, 2 * kwargs['max_score'])
        return kwargs['ax'].plot([], [], label=metric_names[metric])[0], [[], []]
    else:
        x, y = kwargs['history']
        x.append(len(x))
        y.append(kwargs['score'][metric][0])
        kwargs['plot'].set_data(x, y)
        return kwargs['plot'], [x, y]


plotter = {
    "mse": plot_anomaly_metric,
    "psnr": plot_anomaly_metric,
    'ssim': plot_anomaly_metric,
    "input": plot_input_frame,
    "prediction": plot_predicted_frame
}
