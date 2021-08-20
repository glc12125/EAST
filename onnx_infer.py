import onnxruntime
import sys
import cv2
import numpy as np
import lanms
import time
from PIL import Image, ImageDraw


from dataset import get_rotate_mat


def plot_boxes(img, boxes):
    '''plot boxes on image
    '''
    if boxes is None:
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
    return img

def resize_img(img):
    '''resize image to be divisible by 32
    '''
    w, h = img.size
    resize_w = w
    resize_h = h

    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_h = resize_h / h
    ratio_w = resize_w / w

    return img, ratio_h, ratio_w

def is_valid_poly(res, score_shape, scale):
    '''check if the poly in image scope
    Input:
        res        : restored poly in original image
        score_shape: score map shape
        scale      : feature map -> image
    Output:
        True if valid
    '''
    cnt = 0
    for i in range(res.shape[1]):
        if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False

def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    '''restore polys from feature maps in given positions
    Input:
        valid_pos  : potential text positions <numpy.ndarray, (n,2)>
        valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
        score_shape: shape of score map
        scale      : image / feature map
    Output:
        restored polys <numpy.ndarray, (n,8)>, index
    '''
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :] # 4 x N
    angle = valid_geo[4, :] # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0,:] += x
        res[1,:] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
    return np.array(polys), index

def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    '''get boxes from feature map
    Input:
        score       : score map from model <numpy.ndarray, (1,row,col)>
        geo         : geo map from model <numpy.ndarray, (5,row,col)>
        score_thresh: threshold to segment score map
        nms_thresh  : threshold in nms
    Output:
        boxes       : final polys <numpy.ndarray, (n,9)>
    '''
    print("score.shape: {}".format(score.shape))
    print("geo.shape: {}".format(geo.shape))
    if type(score).__name__ != 'ndarray':
        score = score.squeeze(0).cpu().detach().numpy()
        geo = geo.squeeze(0).cpu().detach().numpy()
    score = score[0,:,:]
    xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    return boxes

def resize_boxes(boxes, original_w, original_h, w, h):
	ratio_w = original_w / w
	ratio_h = original_h / h
	for r in range(len(boxes)):
		points = boxes[r]
		print("[DEBUG] detected boxes on resized images: {}".format(points))
		points[0] *= ratio_w
		points[2] *= ratio_w
		points[4] *= ratio_w
		points[6] *= ratio_w
		points[1] *= ratio_h
		points[3] *= ratio_h
		points[5] *= ratio_h
		points[7] *= ratio_h
		print("[DEBUG] resized boxes on original images: {}".format(points))



def detect(session, image_src):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    #image_src = Image.open(img_path)
    #img, ratio_h, ratio_w = resize_img(image_src)
    original_h, original_w = image_src.shape[:2]
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})

    boxes = get_boxes(outputs[0].squeeze(0), outputs[1].squeeze(0))
    #return adjust_ratio(boxes, ratio_w, ratio_h)
    resize_boxes(boxes, original_w, original_h, IN_IMAGE_W, IN_IMAGE_H)
    return boxes, resized

def main(weight_file, image_path, batch_size, IN_IMAGE_H = 704, IN_IMAGE_W = 1280):

    session = onnxruntime.InferenceSession(weight_file)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)
    img = cv2.imread(image_path)
    start = time.time()
    boxes, resized = detect(session, img)
    end = time.time()

    start = time.time()
    for i in range(10):
    	boxes, resized = detect(session, img)
    end = time.time()
    print("time: {}".format((end-start)/10))
    img = Image.open(image_path)
    #img = Image.fromarray(resized, 'RGB')
    plot_img = plot_boxes(img, boxes)
    plot_img.save("result.png")

if __name__ == '__main__':
    print("Loading onnx model ...")
    if len(sys.argv) == 6:
        weight_file = sys.argv[1]
        image_path = sys.argv[2]
        batch_size = int(sys.argv[3])
        IN_IMAGE_H = int(sys.argv[4])
        IN_IMAGE_W = int(sys.argv[5])

        main(weight_file, image_path, batch_size, IN_IMAGE_H, IN_IMAGE_W)
    else:
        print('Please run this way:\n')
        print('  python onnx_infer.py <weight_file> <image_path> <batch_size> <IN_IMAGE_H> <IN_IMAGE_W>')