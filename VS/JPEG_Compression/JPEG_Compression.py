
# coding: utf-8

# In[1]:


import sys
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import copy
import numpy as np
from PIL import Image


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#pylab.rcParams['figure.figsize'] = (20.0, 7.0)


# In[3]:


#этот блок только для отладки
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
DebugPrintList = [
    #'ARRAY',
    #'STEP1',
    #'STEP2',
    #'STEP3',
    #'RESIZE',
    #'STEP4',
    #'STEP5',
    #'STEP6',
    #'STEP7',
    ]

def debug_print(mode, *arguments):
    if (mode in DebugPrintList):
        for arg in arguments:
            print (arg, end = '')
        print ('')
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
              


# In[4]:


# открываем файл

try:
    RGBImage = Image.open("test.bmp")
    #RGBImage = Image.new("RGBA", image.size, (255,255,255,0))
    #RGBImage.paste(image)
    RGBArr = np.array(RGBImage)
except: 
    print("Ошибка при открытии файла. Проверте имя или наличие файла в каталоге с проектом [EXIT]")
    sys.exit()
    
print ("Input BMP Image Size:")
print (RGBImage.size)
#plt.imshow(RGBImage)


# # ШАГ 1. Цветное изображение преобразуется из RGB в представление светимость/цветность (YCbCr)

# In[5]:


DEBUG_MODE = 'ARRAY'
YCbCrImage = RGBImage.convert('YCbCr') 
YCbCrArr = np.array(YCbCrImage)

#этот блок только для отладки
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
if (DEBUG_MODE in DebugPrintList):
    #YCbCrArr = np.array([[[100, 200, 300],[103, 203, 303],[106, 206, 306]],
    #                        [[101, 201, 301],[104, 204, 304],[107, 207, 307]],
    #                        [[102, 202, 302],[105, 205, 305],[108, 208, 308]],
    #                        ])
    YCbCrArr = np.empty(shape = (15,15,3), dtype = int)
    x =0 
    for i in range(15):
        for j in range(15):
            YCbCrArr[i,j,0] = x
            x += 1
    YCbCrArr[:,:,0] = 255
    YCbCrArr[:,:,1] = 0
    YCbCrArr[:,:,2] = 200
    debug_print((DEBUG_MODE,'TEST ARRAY:\r\n',YCbCrArr))
#******************************************************************************
#******************************************************************************
#******************************************************************************
#******************************************************************************
## In[ ]:


#from numpy import unique
#from scipy.stats import entropy as scipy_entropy


#def shannon_entropy(image, base=2):
#    """Calculate the Shannon entropy of an image.
#    The Shannon entropy is defined as S = -sum(pk * log(pk)),
#    where pk are frequency/probability of pixels of value k.
#    Parameters
#    ----------
#    image : (N, M) ndarray
#        Grayscale input image.
#    base : float, optional
#        The logarithmic base to use.
#    Returns
#    -------
#    entropy : float
#    Notes
#    -----
#    The returned value is measured in bits or shannon (Sh) for base=2, natural
#    unit (nat) for base=np.e and hartley (Hart) for base=10.
#    References
#    ----------
#    .. [1] https://en.wikipedia.org/wiki/Entropy_(information_theory)
#    .. [2] https://en.wiktionary.org/wiki/Shannon_entropy
#    Examples
#    --------
#    >>> from skimage import data
#    >>> shannon_entropy(data.camera())
#    7.0479552324230861
#    """

#    _, counts = unique(image, return_counts=True) 
#    return scipy_entropy(counts, base=base)

#def entropy(pk, qk=None, base=None):
#    """Calculate the entropy of a distribution for given probability values.
#    If only probabilities `pk` are given, the entropy is calculated as
#    ``S = -sum(pk * log(pk), axis=0)``.
#    If `qk` is not None, then compute the Kullback-Leibler divergence
#    ``S = sum(pk * log(pk / qk), axis=0)``.
#    This routine will normalize `pk` and `qk` if they don't sum to 1.
#    Parameters
#    ----------
#    pk : sequence
#        Defines the (discrete) distribution. ``pk[i]`` is the (possibly
#        unnormalized) probability of event ``i``.
#    qk : sequence, optional
#        Sequence against which the relative entropy is computed. Should be in
#        the same format as `pk`.
#    base : float, optional
#        The logarithmic base to use, defaults to ``e`` (natural logarithm).
#    Returns
#    -------
#    S : float
#        The calculated entropy.
#    """
#    pk = asarray(pk)
#    pk = 1.0*pk / np.sum(pk, axis=0)
#    if qk is None:
#        vec = entr(pk)
#    else:
#        qk = asarray(qk)
#        if len(qk) != len(pk):
#            raise ValueError("qk and pk must have same length.")
#        qk = 1.0*qk / np.sum(qk, axis=0)
#        vec = rel_entr(pk, qk)
#    S = np.sum(vec, axis=0)
#    if base is not None:
#        S /= log(base)
#    return S


def entr(pk, qk=None, base=None):
    result = - pk * np.log(pk)
    return result



def entropy(pk, qk=None, base=None):
    #pk = np.asarray(pk)
    #pk = 1.0*pk / np.sum(pk, axis=0)
    #if qk is None:
    #    vec = entr(pk)
    #else:
    #    qk = asarray(qk)
    #    if len(qk) != len(pk):
    #        raise ValueError("qk and pk must have same length.")
    #    qk = 1.0*qk / np.sum(qk, axis=0)
    #    vec = rel_entr(pk, qk)
    pk = 1.0*pk / np.sum(pk)
    vec = entr(pk)
    S = np.sum(vec, axis=0)
    if base is not None: S /= np.log(base)
    return S

def shannon_entropy(image, base=2):
    _, counts = np.unique(image, return_counts=True) #определяем количество уникальных значений в массиве
    return entropy(counts, base=base)

TestArr = np.array([
                    [100, 100, 300],
                    [103, 203, 303],
                    [106, 206, 306]
                    ])

H = shannon_entropy(TestArr)

print ("Entropy: ", H)








