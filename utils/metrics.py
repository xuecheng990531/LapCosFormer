import numpy as np

def calculate_sad_mse_mad_whole_img(predict, alpha):
	pixel = predict.shape[0]*predict.shape[1]
	sad_diff = np.sum(np.abs(predict - alpha))/1000
	mse_diff = np.sum((predict - alpha) ** 2)/pixel
	mad_diff = np.sum(np.abs(predict - alpha))/pixel
	return sad_diff, mse_diff, mad_diff	


def compute_gradient_whole_image(pd, gt):
	from scipy.ndimage import gaussian_filter

	pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
	pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
	gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
	gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
	pd_mag = np.sqrt(pd_x**2 + pd_y**2)
	gt_mag = np.sqrt(gt_x**2 + gt_y**2)

	error_map = np.square(pd_mag - gt_mag)
	loss = np.sum(error_map) / 10
	return loss

def compute_connectivity_loss_whole_image(pd, gt, step=0.1):
	from scipy.ndimage import distance_transform_edt
	from skimage.measure import label, regionprops
	h, w = pd.shape
	thresh_steps = np.arange(0, 1.1, step)
	l_map = -1 * np.ones((h, w), dtype=np.float32)
	# lambda_map = np.ones((h, w), dtype=np.float32)
	for i in range(1, thresh_steps.size):
		pd_th = pd >= thresh_steps[i]
		gt_th = gt >= thresh_steps[i]
		label_image = label(pd_th & gt_th, connectivity=1)
		cc = regionprops(label_image)
		size_vec = np.array([c.area for c in cc])
		if len(size_vec) == 0:
			continue
		max_id = np.argmax(size_vec)
		coords = cc[max_id].coords
		omega = np.zeros((h, w), dtype=np.float32)
		omega[coords[:, 0], coords[:, 1]] = 1
		flag = (l_map == -1) & (omega == 0)
		l_map[flag == 1] = thresh_steps[i-1]
		dist_maps = distance_transform_edt(omega==0)
		dist_maps = dist_maps / dist_maps.max()
	l_map[l_map == -1] = 1
	d_pd = pd - l_map
	d_gt = gt - l_map
	phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
	phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
	loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000
	return loss





if __name__=='__main__':
	import torch
	pred=torch.randn(1,1,512,512)
	pred=pred.data.cpu().numpy()
	pred=pred[0,0,:,:]
	true=torch.randn(1,1,512,512)
	true=true.data.cpu().numpy()
	true=true[0,0,:,:]
	mse, sad, mad = calculate_sad_mse_mad_whole_img(pred, true)
	print(mse, sad, mad)
	conn=compute_connectivity_loss_whole_image(pred, true)
	print(conn)
	grad=compute_gradient_whole_image(pred, true)
	print(grad)