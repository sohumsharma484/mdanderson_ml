import scipy.io
import numpy as np
import cv2
import pickle

def load_tissue_imgs(filename):
    img = cv2.imread(filename + '_ISH.jpg')   
    mask = cv2.imread(filename + '_mask.png', cv2.IMREAD_GRAYSCALE)

    return img, mask

def mask_to_color(mask):
    mask = mask.astype(np.uint8)
    color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_img[mask == 0] = [255, 255, 255] # Infiltrating Tumor
    color_img[mask == 1] = [201, 60, 183] # Infiltrating Tumor
    color_img[mask == 2] = [143,217,251] # Perinecrotic zone
    color_img[mask == 3] = [50, 93, 168] # Leading Edge
    color_img[mask == 4] = [16, 46, 199] # Pseudopalisading cells but no visible necrosis
    color_img[mask == 5] = [60, 201, 32] # Cellular Tumor
    color_img[mask == 6] = [0, 0, 0] # Necrosis
    color_img[mask == 7] = [255, 0, 0] # Microvascular proliferation
    color_img[mask == 8] = [255, 138, 5] # Hyperplastic blood vessels
    color_img[mask == 9] = [83,195, 189] # Pseudopalisading cells around necrosis
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) 
    return color_img    


def mIOU(prediction, label, num_classes):
    present_iou_list = list()

    for sem_class in range(1, num_classes):
        pred_inds = (prediction == sem_class)
        target_inds = (label == sem_class)
        if target_inds.sum().item() == 0:
            continue
        else:
            intersection_now = (pred_inds[target_inds]).sum().item()
            union_now = pred_inds.sum().item() + target_inds.sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
    miou = np.mean(present_iou_list)
    return miou

def mdice_coef(prediction, label, num_classes):
    present_dice_list = list()

    for sem_class in range(1, num_classes):
        pred_inds = (prediction == sem_class)
        target_inds = (label == sem_class)
        # Skip calculation if class does not exist in target
        if target_inds.sum().item() == 0:
            continue
        else:
            intersect = np.sum(pred_inds*target_inds)
            total_sum = np.sum(pred_inds) + np.sum(target_inds)
            dice = np.mean(2*intersect/total_sum)
            present_dice_list.append(dice)
    mdice = np.mean(present_dice_list)
    return mdice

