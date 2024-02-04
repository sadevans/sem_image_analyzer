import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.stats import norm as normal_distr
from scipy.stats import poisson, gamma
from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import structural_dissimilarity as dssim



from sklearn.mixture import GaussianMixture


def show_imgs(raw, den, noise, title, el):
    if den is None:
        fig, ax = plt.subplots(figsize=(15,15))

        plt.imshow(raw, cmap='gray')
        plt.title(f'Raw {el} img {title}')
        
    else:
        # ax[1].imshow(den, cmap='gray')
        # ax[1].set_title(f'Denoised {el} img {title}')
        if noise is not None:
            fig, ax = plt.subplots(1,2, figsize=(15,15))

            ax[0].imshow(raw, cmap='gray')
            ax[0].set_title(f'Raw {el} img {title}')

            ax[1].imshow(den, cmap='gray')
            ax[1].set_title(f'Denoised {el} img {title}')

            ax[2].imshow(noise, cmap='gray')
            ax[2].set_title(f'Denoised {el} img {title}')

        if noise is None:
            fig, ax = plt.subplots(1,2, figsize=(15,15))

            ax[0].imshow(raw, cmap='gray')
            ax[0].set_title(f'Raw {el} img {title}')

            ax[1].imshow(den, cmap='gray')
            ax[1].set_title(f'Denoised {el} img {title}')

    plt.show()


x = 100
def plot_rawden_signals(signal_raw, signal_den, name, el, color):

    if color is None:
        color = 'blue'

    if el=='Si':
        el='кремния'
    if el=='PhR' or el=='Ph':
        el='резиста'

    if signal_den is None:
        fig, ax = plt.subplots(figsize=(30,10))
        plt.plot(signal_raw[x,:1000].ravel(), color=color, linewidth=1)
        plt.title(f'Сигнал {el} на изображении {name}')
        plt.xlabel('Пиксели')
        plt.ylabel('Интенсивность')
        plt.text(0, signal_raw[x,:1000].ravel().max(), f'{name}', fontsize=20)
        plt.grid()
    else:
        fig, ax = plt.subplots(3,1, figsize=(30,15))

        ax[0].plot(signal_raw[x,:1000].ravel(), color='blue', linewidth=1)
        ax[0].set_title(f'Сигнал {el} на изображении {name}')
        ax[0].set_xlabel('Пиксели')
        ax[0].set_ylabel('Интенсивность')
        ax[0].grid()

        ax[1].plot(signal_den[x,:1000].ravel(), color='red', linewidth=1)
        ax[1].set_title(f'Чистый сигнал {el} на изображении {name}')
        ax[1].set_xlabel('Пиксели')
        ax[1].set_ylabel('Интенсивность')
        ax[1].grid()

        ax[2].plot(signal_raw[x,:1000].ravel(), color='blue', linewidth=1, label='сигнал шумного изображения')
        ax[2].plot(signal_den[x,:1000].ravel(), color='red', linewidth=1, label='сигнал чистого изображения')
        ax[2].set_title(f'Сигналы на изображении {name}')
        ax[2].set_xlabel('Пиксели')
        ax[2].set_ylabel('Интенсивность')
        ax[2].legend()
        ax[2].grid()

    # plt.grid()

    plt.show()


def plot_noise_signal(signal, name, el, color):
    if color is None:
        color = 'green'

    if el=='Si':
        el='кремния'
    if el=='PhR' or el=='Ph':
        el='резиста'
        
    plt.figure(figsize=(30, 5))

    plt.plot(signal[x, :1000].ravel(), color=color, linewidth=1)
    plt.title(f'Шумный сигнал {el}\n на изображении {name}')
    plt.xlabel('Пиксели')
    plt.ylabel('Интенсивность')
    plt.text(0, signal[x,:1000].ravel().max(), f'{name}', fontsize=20)
    plt.grid()
    plt.show()


def plot_hists_noise(noise, title, el, bins, color):
    if color is None:
        color='green'

    fig, ax = plt.subplots(figsize=(14, 12))
    # bars = plt.hist(raw.ravel())
    plt.hist(noise.ravel(), color=color, bins=bins)
    # plt.text(0, noise[x,:1000].ravel().max(), f'{title}', fontsize=20)
    # plt.bar_label(bars)
    xmin, xmax = plt.xlim()
    plt.title(f'Гистограмма шума {el} изображения {title}')
    plt.xlim(xmin, 255)


def plot_hists(raw, den, noise, bins, title, el, color):
    if color is None:
        color='green'

    if den is None:
        fig, ax = plt.subplots(figsize=(14, 12))
        # bars = plt.hist(raw.ravel())
        plt.hist(raw.ravel(), color=color, bins=bins)
        # plt.bar_label(bars)
        plt.title(f'Hist of {el} raw img {title}')
        # plt.text(0, raw[x,:1000].ravel().max(), f'{title}', fontsize=20)
        plt.xlim(0, 255)

    else:
        if noise is None:
            fig, ax = plt.subplots(1,3, figsize=(14, 12))

            ax[0].hist(raw.ravel(), bins=bins)
            # ax[0].set_title(f'Hist of {el} raw img {title}')
            ax[1].hist(den.ravel(), color='r', bins=bins)
            # ax[1].set_title(f'Hist of {el} denoised img {title}')

            ax[2].hist(raw.ravel(), bins=bins)
            ax[2].hist(den.ravel(), color='r', bins=bins)
            # ax[2].set_title(f'Hist of {el} img {title}')


        if noise is not None:
            fig, ax = plt.subplots(1,4, figsize=(30, 12))

            ax[0].hist(raw.ravel(), bins=bins)
            ax[0].set_title(f'Hist of {el} raw image {title}')

            ax[1].hist(den.ravel(), color='r', bins=bins)
            ax[1].set_title(f'Hist of {el} denoised image {title}')

            ax[2].hist(noise.ravel(), color = color, bins=bins)
            ax[2].set_title(f'Hist of {el} noise in {title}')

            ax[3].hist(raw.ravel(), bins=bins)
            ax[3].hist(den.ravel(), color='r', bins=bins)
            ax[3].set_title(f'Hist of {el} denoised + raw image {title}')

        # ax[4].hist(raw.ravel())
        # ax[4].hist(den.ravel(), color='r')
        # ax[4].hist(noise.ravel(), color = 'g')
        # ax[4].set_title(f'Hist of {el} denoised + raw + noise \nimage {title}')
    plt.grid()
    plt.show()


def check_distr(noise, distr_type, el, name, color=None, bins=50):
    if el=='Si':
        el='кремния'
    if el=='PhR' or el=='Ph':
        el='резиста'


    if color is None:
        color = 'blue'

    df = pd.DataFrame(noise.reshape(-1))
    ax = df.hist(bins=bins, density=True, alpha=0.5, color=color)
        
    if distr_type == 'norm':
        distr_name = 'нормальное'
        distr = stats.norm

    if distr_type == 'gamma':
        distr_name = 'гамма'
        distr = stats.gamma

    if distr_type == 'poisson':
        distr = stats.poisson
        params = distr.pmf

        
    params = distr.fit(noise.reshape(-1))
    mu = noise.reshape(-1).mean()
    std = noise.reshape(-1).std()
    med = np.median(noise.reshape(-1))
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    # print(xmin, xmax)
    # print(noise.min(), noise.max())
    x = np.linspace(xmin, xmax, 100)

    distr_func = distr.pdf(x, *params)

    if distr_type == 'norm':
        plt.plot(x, distr_func, 'b', linewidth=2, label=r'{} $loc = {:.2f}; scale = {:.2f}$'.format(distr_name, *params))
    if distr_type == 'gamma':
        plt.plot(x, distr_func, 'b', linewidth=2, label=r'{} $a = {:.2f}; loc = {:.2f}; scale = {:.2f}$'.format(distr_name, *params))
    if distr_type == 'poisson':
        plt.plot(x, distr_func, 'b', linewidth=2, label=r'{} $mu = {:.2f}$'.format(distr_name, *params))

    plt.plot([mu, mu], [ymin, ymax], 'r', linewidth=1, label=r'$\mu$={:.2f}'.format(mu))
    plt.plot([med, med], [ymin, ymax], 'g', linewidth=1, label=r'median={:.2f}'.format(med))
    plt.title(f'{distr_name} распределение {el} изображение {name}')
    plt.grid()
    plt.legend(loc='best')
    plt.show()

    return params


def qq_plot(noise_dist, name, el, dist_type, dist_params):
    if el=='Si':
        el='кремния'
    if el=='PhR' or el=='Ph':
        el='резиста'
    # Построение Q-Q графика
    if dist_type == 'gamma':
        # params = params
        distr_name = 'гамма'
        stats.probplot(noise_dist.ravel(), dist=dist_type, sparams=dist_params, plot=plt)
        plt.legend(['our distribution', r'{}$a = {:.2f}; loc = {:.2f}; scale = {:.2f}$'.format(distr_name, *dist_params)])
    if dist_type == 'norm':
        distr_name = 'нормальное'
        stats.probplot(noise_dist.ravel(), dist=dist_type, sparams=dist_params, plot=plt)
        plt.legend(['our distribution', r'{}$loc = {:.2f}; scale = {:.2f}$'.format(distr_name, *dist_params)])
        
    plt.title(f"{distr_name} распределение {el} изображение {name}")
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Ordered values")
    plt.grid()
    # plt.xlim(-200, 500)
    # plt.ylim(-200, 500)
    
    plt.show()


def calc_brightness(img):
    average_brightness = img.mean()
    return np.round(average_brightness, 4)


def calc_psnr(noisy_crop, noise):
    clean = noisy_crop - noise
    psnr = cv2.PSNR(clean, noisy_crop)
    return np.round(psnr, 4)


def calc_mse(noisy_crop, noise):
    clean = noisy_crop - noise
    return np.mean((clean - noisy_crop) ** 2)


def interpret_bright(bright):
    if bright < 100:  # Этот порог можно настроить по вашему усмотрению
        return "темное"
    elif bright > 200:
        return "светлое"
    else:
        return "среднее"
    

def interpret_psnr(psnr):
    if psnr > 30:
        return "низкий уровень шума"
    elif psnr > 20:
        return "средний уровень шума"
    else:
        return "высокий уровень шума"
    

def dif_distr_gamma(my_distr, distr_params):
    distr = stats.gamma.rvs(distr_params[0], distr_params[1], distr_params[2], len(my_distr.ravel()))
    return np.round(np.mean(np.abs(my_distr.ravel() - distr)), 4)


def calc_contrast(img):
    contrast = (img.ravel().max() - img.ravel().min()) / (img.ravel().max() + img.ravel().min())
    return np.round(contrast, 4)

def calculate_psnr(noisy, clean):
    # clean = noisy_crop - noise
    psnr = cv2.PSNR(clean, noisy)
    return np.round(psnr, 4)


def calculate_ssim(noisy, clean):
    # clean = noisy_crop - noise
    ssim_index = ssim(clean, noisy, data_range=noisy.max() - noisy.min())
    return np.round(ssim_index, 4)


def calculate_ssim(noisy, clean):
    # clean = noisy_crop - noise
    ssim_index = ssim(clean, noisy, data_range=noisy.max() - noisy.min())
    return np.round(ssim_index, 4)