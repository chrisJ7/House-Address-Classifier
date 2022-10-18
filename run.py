'''
            Imports
'''
import cv2
# import matplotlib.pyplot as plt
import numpy as np

# for CNN model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout
# for VGG model
import tensorflow.keras as keras
'''
            Functions
'''
def import_images(imgs):
    '''
    brings in images
    '''
    images_color = list()
    images = list()
    for img in imgs:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_color.append(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
    return(images_color, images)

def create_CNN_model():
    '''
    Creates a CNN model
    '''
    model = Sequential()
    # layer 1
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     activation='relu',
                     input_shape=[32, 32, 1]))
    # layer 2
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'))
    # dropout prevents overfitting
    model.add(Dropout(0.5))
    # Layer 2 can be re-pasted for a deeper network

    # Flattening layer
    model.add(Flatten())

    # Dense layer
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=11, activation='softmax')) # bec/ 11 classes, sigma might work too

    # learning rate = 0.001
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return(model)

def create_VGG_model():
    '''
    Creates a VGG16 model
    '''
    vgg16_model = keras.applications.vgg16.VGG16()

    model = Sequential()
    for layer in vgg16_model.layers[:-1]: # remove last "Dense" layer that classifies 1/1000
        model.add(layer)

    # freezes all layers
    for layer in model.layers:
        layer.trainable = False

    # Dense layer
    model.add(Dense(11, activation='softmax')) # bec/ 11 classes, sigma might work too

    # learning rate = 0.001
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return(model)
'''
            Inline Code
'''
### ### 1) import the images and convert them to grayscale
imgs = ['55.png', '201.png', '237.png', '280.png', '721.png', '3822.png', '438.png', '26.png']
imgs = ['201.png', '237.png', '280.png', '3822.png', '438.png']

images_color, images = import_images(imgs)
print('### Part1/5 Completed')

### ### 2) attempted MSER digit box finder
# for img, img_c in zip(images, images_color):
#     print('img.shape:', img.shape)
#
#     vis = img_c.copy()
#     mser = cv2.MSER_create(_delta=3,
#                            _min_area=400,
#                            _max_area=500,
#                            _max_variation=0.075)
#
#     regions, _ = mser.detectRegions(img)
#
#     hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
#     print('{} regions of interest'.format(len(hulls)))
#
#     cv2.polylines(vis, hulls, 1, (0, 255, 0), thickness=5)
#
#     plt.imshow(vis)

### ### 2-alt) forced digit box finder
img_nums = list()
urs = list()
lls = list()

vis_ = list()
i = 1
for img, img_c in zip(images, images_color):
    vis = img_c.copy()

    if i == 1:
            box_size = 50
            ur = (200, 40)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

            ur = (200, 80)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

            ur = (200, 120)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

    if i == 2:
            box_size = 100
            ur = (200, 250)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

            ur = (270, 250)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

            ur = (320, 250)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

    if i == 3:
            box_size = 120
            ur = (150, 60)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

            ur = (150, 170)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

            ur = (150, 280)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

    if i == 4:
            box_size = 50
            ur = (215, 195)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

            ur = (250, 200)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

            ur = (275, 195)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

            ur = (305, 195)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

    if i == 5:
            box_size = 32
            ur = (250, 105)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

            ur = (275, 105)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

            ur = (295, 105)
            ll = (ur[0]+box_size, ur[1]+box_size)
            cv2.rectangle(vis, pt1=ur, pt2=ll, color=(0, 255, 0), thickness=2)
            urs.append(ur)
            lls.append(ll)
            img_num = img[ur[1]:ll[1], ur[0]:ll[0]]
            img_nums.append(img_num)

#     plt.imshow(vis)
    vis_.append(vis)
    i+=1
print('### Part2/5 Completed')

### ### 3) select classifier and reshape imgs
selected_classifier = 'CNN'
# selected_classifier = 'VGG'

if selected_classifier == 'CNN':
    resized_list = list()
    for img in img_nums:
        resized = cv2.resize(img, (32, 32))
        resized_list.append(resized)
    num_images = len(resized_list)
    reshaped = np.reshape(resized_list,(num_images,32,32,1)).astype('float64')
elif selected_classifier == 'VGG':
    resized_list = list()
    for img in img_nums:
        resized = cv2.resize(img, (224, 224))
        resized_list.append(resized)
    num_images = len(resized_list)
    reshaped = np.reshape(resized_list,(num_images,224,224,1)).astype('float64')
    reshaped_2 = np.zeros((num_images, 224, 224, 3), dtype='uint8')
    for i in range(reshaped.shape[0]):
        img = reshaped[i, :, :, 0]
        reshaped_2[i,:,:,0] = img
        reshaped_2[i,:,:,1] = img
        reshaped_2[i,:,:,2] = img
    reshaped = reshaped_2.astype('float32')
print('### Part3/5 Completed')

### ### 4) create classifier, train with pre-trained weights, and classify images
if selected_classifier == 'CNN':
    model = create_CNN_model()
    model.load_weights('CNN_weights')
elif selected_classifier == 'VGG':
    model = create_VGG_model()
    model.load_weights('VGG_weights')

y_pred = model.predict_classes(x=reshaped)
print('### Part4/5 Completed')

### ### 5) label images
vis_2 = list()
for im in vis_:
    i = np.copy(im)
    vis_2.append(i)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 255, 0)
thickness = 2

labeled_images = list()
im = 1
i = 0
for vis in vis_2:
    if im == 1:
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1

    if im == 2:
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1

    if im == 3:
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1

    if im == 4:
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1

    if im == 5:
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1
        text = str(y_pred[i])
        org = (int((lls[i][0]+urs[i][0])/2)-20, urs[i][1]-10)
        image = cv2.putText(vis, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        i += 1

    im += 1
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     plt.imshow(image)
    cv2.imwrite('graded_images/'+str(im-1)+'.png', image)

print('### Part5/5 Completed')
