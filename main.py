import numpy as np


class DeepBackPropagation:
    def __init__(self, mask) -> None:
        self.mask = mask

    def convolute_img(self):
        self.mask_normalized = self.mask / np.sum(self.mask)
        column, row = self.img.shape
        self.img_convolution = np.zeros((column - 2, row - 2))
        for index_column in range(0, column - 2):
            for index_row in range(0, row - 2):
                next_column = index_column + 3
                next_row = index_row + 3
                sub_img = self.img[index_column:next_column,
                                   index_row:next_row]
                sub_img = (sub_img * self.mask_normalized) / np.max(self.img)
                self.img_convolution[index_column, index_row] = sub_img.sum()
        column_convolution, row_convolution = self.img_convolution.shape
        self.number_neural_network_1 = column * row
        self.number_weight_1 = column_convolution * row_convolution
        self.number_neural_network_2 = column * row
        self.number_weight_2 = column * row
        self.layer_1 = np.random.randn(
            self.number_neural_network_1, self.number_weight_1)
        self.bias_1 = np.ones(self.number_neural_network_1)
        self.output_1 = np.zeros(self.number_neural_network_1)
        self.layer_2 = np.random.randn(
            self.number_neural_network_2, self.number_weight_2)
        self.bias_2 = np.ones(self.number_neural_network_2)
        self.output_2 = np.zeros(self.number_neural_network_2)

    def set_image(self, img):
        self.img = img

    def feed_forward(self):
        self.convolute_img()
        self.img_convolution_vector = self.img_convolution.ravel()
        for index_mask in range(0, self.number_neural_network_1):
            sum_weight = self.img_convolution_vector * self.layer_1[index_mask]
            self.output_1[index_mask] = self.bias_1[index_mask] + \
                sum_weight.sum()
        self.output_1 = 1/(1 + np.exp(-self.output_1))
        for index_mask in range(0, self.number_neural_network_2):
            sum_weight = self.output_1 * self.layer_2[index_mask]
            self.output_2[index_mask] = self.bias_2[index_mask] + \
                sum_weight.sum()
        self.output_2 = 1/(1 + np.exp(-self.output_2))
        self.activation = np.clip(np.round(self.output_2), 0, 1)
        return self.output_2, self.activation

    def back_propagation(self):
        bias_derivate_1 = np.zeros(self.number_neural_network_1)
        # bias_derivate_1 = - (2 * ((self.target - self.activation) *
        #                         np.exp(-self.output_1))) / np.square(1 + np.exp(-self.output_1))
        bias_derivate_2 = - (2 * ((self.target - self.activation) *
                                np.exp(-self.output_2))) / np.square(1 + np.exp(-self.output_2))
        weight_derivate_1 = np.zeros(
            (self.number_neural_network_1, self.number_weight_1))
        # for index_mask in range(0, self.number_neural_network_1):
        #     weight_derivate_1[index_mask] = - ((2 * ((self.target - self.activation) * np.exp(-self.output_1)))
        #                                      * self.img_convolution_vector) / np.square(1 + np.exp(-self.output_1))
        
        weight_derivate_2 = np.ones(
            (self.number_neural_network_2, self.number_weight_2))
        for index_mask in range(0, self.number_neural_network_2):
            weight_derivate_2[index_mask] = - ((2 * ((self.target - self.activation) * np.exp(-self.output_2)))
                                             * self.output_1) / np.square(1 + np.exp(-self.output_2))
            
        mask_vector = np.concatenate(self.mask)
        row_mask = mask_vector.shape[0]
        mask_derivate = np.zeros(row_mask)
        for index_mask in range(0, row_mask):
            double_sum = 0
            for index_neural in range(0, self.number_neural_network_1):
                sum = self.layer_1[index_neural] * \
                    self.img_convolution_vector * mask_vector[index_mask]
                double_sum = double_sum + sum.sum()
            mask_error = (((self.target - self.activation) * np.exp(-self.output_1)) /
                          np.square(1 + np.exp(-self.output_1))) * double_sum
            mask_derivate[index_mask] = - 2 * mask_error.sum()
        mask_derivate_new = np.reshape(mask_derivate, (3, 3))
        # justing weight
        self.bias_1 = self.bias_1 + bias_derivate_1
        for index_neural in range(0, self.number_neural_network_1):
            self.layer_1[index_neural] = self.layer_1[index_neural] + \
                weight_derivate_1[index_neural]
        self.bias_2 = self.bias_2 + bias_derivate_2
        for index_neural in range(0, self.number_neural_network_2):
            self.layer_2[index_neural] = self.layer_2[index_neural] + \
                weight_derivate_2[index_neural]
        # print(mask_derivate_new)
        self.mask = self.mask + mask_derivate_new
        # print(self.mask)

    def train(self, target):
        self.target = target
        self.feed_forward()
        self.back_propagation()

from PIL import Image
from matplotlib import pyplot

percentage_to_reducer = .3
# image = Image.open('./DRIVE/Original/01_test.tif').convert('L')
# width, height = image.size
# resized_dimensions = (int(width * percentage_to_reducer), int(height * percentage_to_reducer))
# resized = image.resize(resized_dimensions)
# img = np.array(resized)


# image_target = Image.open('./DRIVE/Groundtruth/01_manual1.tif').convert('L')
# width, height = image_target.size
# resized_dimensions = (int(width * percentage_to_reducer), int(height * percentage_to_reducer))
# resized = image_target.resize(resized_dimensions)

mask = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])

# target = np.array(resized).ravel()

DBP = DeepBackPropagation(mask=mask)
for i in range(1):
    for ii in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']:
        image = Image.open('./training/input/{}_test.tif'.format(ii)).convert('L')
        width, height = image.size
        resized_dimensions = (int(width * percentage_to_reducer), int(height * percentage_to_reducer))
        resized = image.resize(resized_dimensions)
        img = np.array(resized)

        image_target = Image.open('./training/target/{}_manual1.tif'.format(ii)).convert('L')
        width, height = image_target.size
        resized_dimensions = (int(width * percentage_to_reducer), int(height * percentage_to_reducer))
        resized = image_target.resize(resized_dimensions)
        target = np.array(resized).ravel()
        DBP.set_image(img=img)
        output, activation = DBP.feed_forward()
    



        print(target)
        print(output)
        print("Loss: " + str(np.mean(np.square(target - output))))
        DBP.train(target=target)
