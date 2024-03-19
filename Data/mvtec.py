import os
import random
from utils import files_to_tensor, cls_files, get_command_line_parser
from category_split import *
class metaMvTec():
    """
    划分MVTec数据集
    """
    def __init__(self, root = 'D:\\datasets\\meta-learning\\mvtec'):
        self.root = root
        self.remove_cls = 'bottle'
        self.n_way = 5
        self.k_shot = 1
        self.k_query = 5
        self.category = [name for name in os.listdir(self.root)]
        self.category.remove(self.remove_cls)
        self.cls_to_files = cls_files(self.category, self.root)


    def test(self):
        return self.sample()

    def sample(self, mode = 'train'):
        target_to_files = {}
        data_dict = {}

        if 'train' == mode:
            for cls in train_cls:
                target_to_files[cls] = self.cls_to_files[cls]
        elif 'val' == mode:
            for cls in val_cls:
                target_to_files[cls] = self.cls_to_files[cls]
        elif 'test' == mode:
            for cls in test_cls:
                target_to_files[cls] = self.cls_to_files[cls]
        else:
            raise "input error!"

        random_way = random.sample(target_to_files.keys(), self.n_way)

        label = [i for i in range(5)]
        random.shuffle(label)   # random label

        for l, way in zip(label, random_way):
            data_dict[l] = random.sample(target_to_files[way],self.k_shot + self.k_query)

        support_list = [(i, file) for i, files in data_dict.items() for file in files[:self.k_shot]]
        query_list = [(i, file) for i, files in data_dict.items() for file in files[self.k_shot:]]
        random.shuffle(support_list), random.shuffle(query_list)
        return support_list, query_list





if __name__ == "__main__":
    dataset = metaMvTec()
    support_list, query_list = dataset.sample(mode='train')
    # val_data = dataset.sample(mode='val')
    # test_data = dataset.sample(mode='test')
    a = 1
    # test_support, test_query = Dataset.sample(is_test=True)
    # support, query = Dataset.sample(is_test=False)
    #  = Dataset.sample_test()
    # support_data, query_data = Dataset.sample_test()





