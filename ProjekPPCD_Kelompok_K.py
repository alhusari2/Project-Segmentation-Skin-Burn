import cv2
import numpy as np
import time
import scipy.ndimage
import skimage.filters
import sklearn.metrics
import tkinter as tk
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog as fd


def _assert_valid_lists(groundtruth_list, predicted_list):
    assert len(groundtruth_list) == len(predicted_list)
    for unique_element in np.unique(groundtruth_list).tolist():
        assert unique_element in [0, 1]


def _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == np.unique(predicted_list).tolist() == [1]


def _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == np.unique(predicted_list).tolist() == [0]


def get_confusion_matrix_elements(groundtruth_list, predicted_list):
    """
    Return confusion matrix elements covering edge cases
    :param groundtruth_list list of groundtruth elements
    :param predicted_list list of predicted elements
    :return returns confusion matrix elements i.e TN, FP, FN, TP in that
    order and as floats
    returned as floats to make it feasible for float division for further
    calculations on them
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    if _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = 0, 0, 0, np.float64(len(groundtruth_list))
    elif _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = np.float64(len(groundtruth_list)), 0, 0, 0
    else:
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(groundtruth_list, predicted_list).ravel()
        tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)
    return tn, fp, fn, tp


def _all_class_0_predicted_as_class_1(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == [0] and np.unique(predicted_list).tolist() == [1]


def _all_class_1_predicted_as_class_0(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == [1] and np.unique(predicted_list).tolist() == [0]


def _mcc_denominator_zero(tn, fp, fn, tp):
    # _assert_valid_lists(groundtruth_list, predicted_list)
    return (tn == 0 and fn == 0) or (tn == 0 and fp == 0) or (tp == 0 and fp == 0) or (tp == 0 and fn == 0)


def get_f1_score(groundtruth_list, predicted_list):
    """
    Return f1 score covering edge cases
    :param groundtruth_list list of groundtruth elements
    :param predicted_list list of predicted elements
    :return returns f1 score
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)
    if _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        f1_score = 1
    elif _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        f1_score = 1
    else:
        f1_score = (2 * tp) / ((2 * tp) + fp + fn)
    return f1_score


def get_mcc(groundtruth_list, predicted_list):
    """
    Return mcc covering edge cases
    :param groundtruth_list list of groundtruth elements
    :param predicted_list list of predicted elements
    :return returns mcc
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)
    if _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = -1
    elif _all_class_0_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        mcc = -1
    elif _mcc_denominator_zero(tn, fp, fn, tp) is True:
        mcc = -1
    else:
        mcc = ((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return mcc


def get_accuracy(groundtruth_list, predicted_list):
    """
    Return accuracy
    :param groundtruth_list list of elements
    :param predicted_list list of elements
    :return returns accuracy
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    return accuracy


def get_validation_metrics(groundtruth_list, predicted_list):
    """
    Return validation metrics dictionary with accuracy, f1 score, mcc after
    comparing ground truth and predicted image
    :param groundtruth_list list of elements
    :param predicted_list list of elements
    :return returns a dictionary with accuracy, f1 score, and mcc as keys
    one could add other stats like FPR, FNR, TP, TN, FP, FN etc
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    validation_metrics = {}
    validation_metrics["accuracy"] = get_accuracy(groundtruth_list, predicted_list)
    validation_metrics["f1_score"] = get_f1_score(groundtruth_list, predicted_list)
    validation_metrics["mcc"] = get_mcc(groundtruth_list, predicted_list)
    return validation_metrics


def select_image():
    global panelA, panelB
    path = fd.askopenfilename()
    if len(path) > 0:
        image_asli = cv2.imread(path)
        hasil_aktual = cv2.imread('img1002.png')
        gamma = 1.1
        img = np.clip(np.power(image_asli, gamma), 0, 255).astype('uint8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('koreksi gamma', img)
        br, kl = len(gray), len(gray[0])
        sig = 1.5
        img_s = cv2.GaussianBlur(gray, (15, 15), sig)

        def grad(x):
            return np.array(np.gradient(x))

        def norm(x, axis=0):
            return np.sqrt(np.sum(np.square(x), axis=axis))

        def levelset(phi_0, g, tms, it):
            for k in range(it):
                phi_0 = phi_0 + tms * g * norm(grad(phi_0))
            return phi_0

        def stopping_fun(x):
            return 1. / (1. + norm(grad(x)) ** 2)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
        img_vect = img_s.reshape((-1, 3))
        img_vect = np.float32(img_vect)

        ret, label, centroids = cv2.kmeans(img_vect, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centroids = np.uint8(centroids)
        img_kmeans = centroids[label.flatten()]
        img_kmeans = img_kmeans.reshape(img_s.shape)

        tresh0 = lambda x: 255 if x <= 2 else 0
        tresh = np.vectorize(tresh0)
        tms = 0.2
        it_in = 15
        it_out = 25
        g = stopping_fun(img_kmeans)
        c0 = 2
        phi = c0 * np.ones((br, kl))
        phi[5:-5, 5:-5] = -c0


        for i in range(it_out):
            phi = levelset(phi, g, tms, it_in)
            gbr = tresh(phi)
            cv2.imshow('Iterasi', np.uint8(gbr))
            cv2.waitKey(1)
            time.sleep(0.1)
            result = np.zeros([br, kl, 3])
            segmentasi = np.zeros([br, kl])
            for j in range(br):
                for k in range(kl):
                    segmentasi[j, k] = gbr[j, k]
                    if gbr[j, k] == 255:
                        gbr[j, k] = 0
                    elif gbr[j, k] == 0:
                        gbr[j, k] = 255
            for j in range(br):
                for k in range(kl):
                    if gbr[j, k] == 0:
                        result[j, k, :] = image_asli[j, k, :]
        cv2.destroyAllWindows()

        csg = np.uint8(segmentasi)
        cv2.imshow('Citra Gamma', img)
        cv2.imshow('Citra asal', image_asli)
        cv2.imshow('Citra level set', csg)
        cv2.imshow('K-means', cv2.cvtColor(img_kmeans, cv2.COLOR_BGR2RGB))
        cv2.imshow('Segmentasi Kulit', np.uint8(result))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        Cropped = np.uint8(result)
        image = cv2.cvtColor(image_asli, cv2.COLOR_BGR2RGB)
        color_converted = cv2.cvtColor(Cropped, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        segmented = Image.fromarray(color_converted)
        image = ImageTk.PhotoImage(image)
        segmented = ImageTk.PhotoImage(segmented)
        # EVALUASI
        groundtruth_scaled = result // 255
        predicted_scaled = image_asli // 255

        groundtruth_list = (groundtruth_scaled).flatten().tolist()
        predicted_list = (predicted_scaled).flatten().tolist()
        accuration = get_validation_metrics(groundtruth_list, predicted_list)
        print("akurasi =", accuration)
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        cm = confusion_matrix(groundtruth_list, predicted_list)
        print(cm)
        print(classification_report(groundtruth_list, predicted_list))
        from sklearn.metrics import accuracy_score
        print('Model accuracy score: {0:0.4f}'.format(accuracy_score(groundtruth_list, predicted_list)))

        if panelA is None or panelB is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            panelB = Label(image=segmented)
            panelB.image = segmented
            panelB.pack(side="right", padx=10, pady=10)
        else:
            panelA.configure(image=image)
            panelB.configure(image=segmented)
            panelA.image = image
            panelB.image = segmented

root = Tk()
panelA = None
panelB = None

btn = Button(root, text="Select an image", command=select_image)
btn.pack(fill="both", expand="yes", padx="10", pady="10")
b2 = Button(root, text = "Exit", command = root.destroy)
b2.pack()

root.mainloop()