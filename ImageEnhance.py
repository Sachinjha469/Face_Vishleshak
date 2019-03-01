import cv2
from PIL import Image, ImageEnhance

new_path='test/a.jpg'

def histogram_equalize(path_to_image):
    img = cv2.imread(path_to_image)
    img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(new_path,hist_equalization_result)
    return new_path


def increase_brightness(path_to_image):
    im = Image.open(path_to_image)
    enhancer = ImageEnhance.Brightness(im)
    enhanced_im = enhancer.enhance(1.8)
    enhanced_im.save(new_path)
    return new_path

def avg_brightness(path_to_image):
	im = Image.open(path_to_image)
	enhancer = ImageEnhance.Brightness(im)
	enhanced_im = enhancer.enhance(1.8)
	enhanced_im=Image.blend(im,enhanced_im,1.0/float(2))
	enhanced_im.save(new_path)
	return new_path

