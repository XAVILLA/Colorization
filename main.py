import numpy as np
import sys
import skimage as sk
import skimage.io as skio
import time

def ssd(im1, im2):
    return np.sum(np.sum((im1-im2)**2))

def features(img, direction):
    if direction == 'v':
        edge = img[:, 2:] - img[:, :-2]
    elif direction == 'h':
        edge = img[2:, :] - img[:-2,:]
    return edge

def get_histogram(image, bins):
    histogram = np.zeros(bins)
    for pixel in image:
        histogram[pixel] += 1
    return histogram

def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

def equal_hist(img):
    flat = img.flatten()*255

    intarray = flat.astype(np.int)
    hist = get_histogram(intarray, 256)
    cs = cumsum(hist)
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    cs = nj / N
    img_new = cs[intarray]
    img_new = (np.reshape(img_new, img.shape)/255).astype(np.float)
    return img_new

def displace(im1, im2, x_range, y_range, feature):
    if feature == 'edge':
        im1h = features(im1, 'h')
        im1v = features(im1, 'v')
        im2h = features(im2, 'h')
        im2v = features(im2, 'v')

        x_search_range = np.arange(-x_range, x_range+1)
        y_search_range = np.arange(-y_range, y_range+1)
        best_fit = [0, 0]
        err_track = ssd(im1h, im2h) + ssd(im1v, im2v)
        for x in x_search_range:
            for y in y_search_range:
                shiftim2h = np.roll(np.roll(im2h, x, axis = 0), y, axis = 1)
                shiftim2v = np.roll(np.roll(im2v, x, axis = 0), y, axis = 1)
                err = ssd(im1h, shiftim2h) + ssd(im1v, shiftim2v)
                if (err < err_track):
                    err_track = err
                    best_fit = [x, y]
        return best_fit
    elif feature == 'raw':
        x_search_range = np.arange(-x_range, x_range+1)
        y_search_range = np.arange(-y_range, y_range+1)
        best_fit = [0, 0]
        err_track = ssd(im1, im2)
        for x in x_search_range:
            for y in y_search_range:
                shiftim2 = np.roll(np.roll(im2, x, axis = 0), y, axis = 1)
                err = ssd(im1, shiftim2)
                if (err < err_track):
                    err_track = err
                    best_fit = [x, y]
        return best_fit

def slimg(im):
    length = im.shape[0]
    height = im.shape[1]
    cutlen = int(np.floor(length*0.08))
    cuthei = int(np.floor(height*0.08))
    cuthor = im[cutlen: -cutlen]
    cutver = cuthor[:, cuthei: -cuthei]
    return cutver

def recur_colorize(imname, feature, crop):
	start_time = time.time()
	if feature == 'raw':
		printfeature = 'raw pixel'
	else:
		printfeature = 'edge detection'
	print("Colorizing " + imname + " by aligning channels with " + printfeature + '.')
	img = skio.imread(imname)
	img = sk.img_as_float(img)
	height = np.floor(img.shape[0] / 3.0).astype(np.int)
	blue = slimg(img[:height])
	green = slimg(img[height: 2*height])
	red = slimg(img[2*height: 3*height])

	g_dis, r_dis = recur_displace(blue, green, red, feature)
	print('Green:', g_dis, ' Red:', r_dis)
	green_shift = np.roll(np.roll(green, g_dis[0], axis=0), g_dis[1], axis = 1)
	red_shift = np.roll(np.roll(red, r_dis[0], axis=0), r_dis[1], axis = 1)
	if crop:
		cropped = auto_cropping(blue, green_shift, red_shift, g_dis, r_dis)
		result = np.dstack([cropped[2], cropped[1], cropped[0]])
	else:
		result = np.dstack([red_shift, green_shift, blue])
	print("--- %s seconds ---" % (time.time() - start_time))
	return result, g_dis, r_dis

def recur_displace(b_chan, g_chan, r_chan, feature):
    assert b_chan.shape == g_chan.shape == r_chan.shape
    ratio = 2
    rang = 5
    img_size = min(b_chan.shape)
    if img_size <= 100:
        return [displace(b_chan, g_chan, rang, rang, feature), displace(b_chan, r_chan, rang, rang, feature)]
    else:
        b_zoom, g_zoom, r_zoom = b_chan[::ratio,:][:,::ratio],g_chan[::ratio,:][:,::ratio],r_chan[::ratio,:][:,::ratio]
        g_dis_child, r_dis_child = recur_displace(b_zoom, g_zoom, r_zoom, feature)
        g_dis_child = [i*ratio for i in g_dis_child]
        r_dis_child = [i*ratio for i in r_dis_child]
        green_1 = np.roll(np.roll(g_chan, g_dis_child[0], axis=0), g_dis_child[1], axis = 1)
        red_1 = np.roll(np.roll(r_chan, r_dis_child[0], axis=0), r_dis_child[1], axis = 1)
        g_dis_cur = displace(b_chan, green_1, 2, 2, feature)
        r_dis_cur = displace(b_chan, red_1, 2, 2, feature)
        g_final = [g_dis_child[0] + g_dis_cur[0], g_dis_child[1] + g_dis_cur[1]]
        r_final = [r_dis_child[0] + r_dis_cur[0], r_dis_child[1] + r_dis_cur[1]]

        return g_final, r_final

def colorize_with_hist(imname, feature, crop):
	img, g_dis, r_dis = recur_colorize(imname, feature, crop)
	print("Finished aligning the channels, performing histogram equalization.")
	new_img = equal_hist(img)
	return new_img, g_dis, r_dis

def execute(imname, feature, eh, crop):
	assert eh == 'yes' or eh == 'no'
	assert feature == 'edge' or feature == 'raw'
	assert crop == 'yes' or 'no'
	crop = (crop == 'yes')
	if crop:
		strcrop = '_crop'
	else:
		strcrop = ''
	if imname[-4:] == '.tif':
		img_format = '.tif'
		prename = imname[:-4]
	else:
		img_format = '.jpg'
		prename = imname[:-5]
	if feature == 'raw':
		if eh == 'no':
			new_img, g_dis, r_dis = recur_colorize(imname, feature, crop)
			fname = 'out_' + prename + '_raw_pixel_matching' + 'G[' + str(g_dis[0])+',' + str(g_dis[1]) + ']' + 'R[' + str(r_dis[0])+',' + str(r_dis[1]) + ']' +strcrop+'.jpg'
			skio.imsave(fname, new_img)
		else:
			print("Please use edge detection alignment feature for equalize histogram feature")
	else:
		if eh == 'no':
			new_img, g_dis, r_dis = recur_colorize(imname, feature, crop)
			fname = 'out_' + prename + '_edge_matching' + 'G[' + str(g_dis[0])+',' + str(g_dis[1]) + ']' + 'R[' + str(r_dis[0])+',' + str(r_dis[1]) + ']' +strcrop+'.jpg'
			skio.imsave(fname, new_img)
		else:
			new_img, g_dis, r_dis = colorize_with_hist(imname, feature, crop)
			fname = 'out_' + prename + ' edge and equal-hist' + 'G[' + str(g_dis[0])+',' + str(g_dis[1]) + ']' + 'R[' + str(r_dis[0])+',' + str(r_dis[1]) + ']' +strcrop+'.jpg'
			skio.imsave(fname, new_img)


def auto_cropping(b_chan, g_chan, r_chan, g_dis, r_dis):
	print('Auto cropping!')
	g_x, g_y  = g_dis
	r_x, r_y  = r_dis
	crop_on_x = crop_amount(g_x, r_x)
	crop_on_y = crop_amount(g_y, r_y)
	b_crop = b_chan[crop_on_x[0]:crop_on_x[1],:][:, crop_on_y[0]:crop_on_y[1]]
	g_crop = g_chan[crop_on_x[0]:crop_on_x[1],:][:, crop_on_y[0]:crop_on_y[1]]
	r_crop = r_chan[crop_on_x[0]:crop_on_x[1],:][:, crop_on_y[0]:crop_on_y[1]]
	return b_crop, g_crop, r_crop


def find_sign(num):
	if num > 0:
		return 1
	elif num == 0:
		return 0
	else:
		return -1

def crop_amount(x1, x2):
	x1_sign, x2_sign = find_sign(x1), find_sign(x2)
	if x1_sign == -1:
		if x2_sign == -1:
			return [0, min(x1, x2)]
		elif x2_sign == 0:
			return [0, x1]
		else:
			return [x2, x1]
	elif x1_sign == 0:
		if x2_sign == -1:
			return [0, x2]
		elif x2_sign == 0:
			return [0, -1]
		else:
			return [x2, -1]
	elif x1_sign == 1:
		if x2_sign == -1:
			return [x1, x2]
		elif x2_sign == 0:
			return [x1, -1]
		else:
			return [max(x1, x2), -1]

if len(sys.argv) > 0 and sys.argv[0] == 'main.py':
	imname = sys.argv[1].split('/')[-1]
	feature_choice = sys.argv[2]
	eh = sys.argv[3]
	auto_crop = sys.argv[4]
	execute(imname, feature_choice, eh, auto_crop)



























