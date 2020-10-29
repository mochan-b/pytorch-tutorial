
import pytest

from imagenet_categories.get_imagenet_category import ImageNetCategory

def test_imagenet_category():
    cats = ImageNetCategory()
    assert cats.get_imagenet_category(7) == 'cock'
    assert cats.get_imagenet_category(312) == 'cricket'