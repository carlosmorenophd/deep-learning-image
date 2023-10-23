import numpy as np


class DeepBackPropagation:
    def __init__(self, mask, img) -> None:
        self.mask = mask
        self.img = img

    def convoluniate_img(self):
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
        self.number_neural_network = column_convolution * row_convolution
        self.number_weight = column_convolution * row_convolution
        self.layer_1 = np.random.randn(
            self.number_neural_network, self.number_weight)
        self.bias_1 = np.ones(self.number_neural_network)
        self.output_1 = np.zeros(self.number_neural_network)

    def feed_foward(self):
        self.convoluniate_img()
        self.img_convolution_vector = self.img_convolution.ravel()
        for index_mask in range(0, self.number_neural_network):
            sum_weight = self.img_convolution_vector * self.layer_1[index_mask]
            self.output_1[index_mask] = self.bias_1[index_mask] + \
                sum_weight.sum()
        self.activation = 1/(1 + np.exp(-self.output_1))
        self.activation_round = np.clip(np.round(self.activation), 0, 1)
        return self.activation, self.activation_round

    def back_propagation(self):
        bias_derivate = - (2 * ((self.target - self.activation) *
                                np.exp(-self.output_1))) / np.square(1 + np.exp(-self.output_1))
        weight_derivate = np.ones(
            (self.number_neural_network, self.number_weight))
        for index_mask in range(0, self.number_neural_network):
            weight_derivate[index_mask] = - ((2 * ((self.target - self.activation) * np.exp(-self.output_1)))
                                             * self.img_convolution_vector) / np.square(1 + np.exp(-self.output_1))
        mask_vector = np.concatenate(self.mask)
        row_mask = mask_vector.shape[0]
        mask_derivate = np.zeros(row_mask)
        for index_mask in range(0, row_mask):
            double_sum = 0
            for index_neural in range(0, self.number_neural_network):
                sum = self.layer_1[index_neural] * \
                    self.img_convolution_vector * mask_vector[index_mask]
                double_sum = double_sum + sum.sum()
            mask_error = (((self.target - self.activation) * np.exp(-self.output_1)) /
                          np.square(1 + np.exp(-self.output_1))) * double_sum
            mask_derivate[index_mask] = - 2 * mask_error.sum()
        mask_derivate_new = np.reshape(mask_derivate, (3, 3))
        # ajusting weight
        self.bias_1 = self.bias_1 + bias_derivate
        for index_neural in range(0, self.number_neural_network):
            self.layer_1[index_neural] = self.layer_1[index_neural] + \
                weight_derivate[index_neural]
        print(mask_derivate_new)
        self.mask = self.mask + mask_derivate_new
        print(self.mask)

    def train(self, target):
        self.target = target
        self.feed_foward()
        self.back_propagation()


img = np.array([
    [2, 2, 6, 4, 8, 5, 4, 4,],
    [1, 1, 9, 3, 3, 5, 5, 7,],
    [0, 0, 0, 6, 0, 4, 9, 8,],
    [3, 3, 8, 5, 6, 7, 7, 5,],
    [4, 2, 5, 7, 1, 6, 0, 6,],
    [1, 2, 7, 0, 4, 0, 2, 1,],
    [8, 6, 7, 2, 6, 8, 1, 0,],
    [8, 9, 0, 1, 4, 2, 1, 2,],
])
mask = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])
target = np.ones(36)
DBP = DeepBackPropagation(img=img, mask=mask)
for i in range(10):
    activation, activation_round = DBP.feed_foward()
    print("Loss: " + str(np.mean(np.square(target - activation_round))))
    DBP.train(target=target)
