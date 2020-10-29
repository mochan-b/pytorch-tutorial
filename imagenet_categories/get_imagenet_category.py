
import yaml
import os

class ImageNetCategory:
    def __init__(self, path='imagenet_categories'):
        with open(os.path.join(path, 'imagenet1000_clsidx_to_labels.txt')) as f:
            self.data = yaml.load(f, Loader=yaml.FullLoader)

    def get_imagenet_category(self, cat_id):
        return self.data[cat_id]