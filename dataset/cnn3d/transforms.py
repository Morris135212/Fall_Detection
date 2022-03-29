import torchvision.transforms as transforms


DATA_SHAPE = (112, 112)


class Padding:
    def __init__(self, size):
        self.w = size[1]
        self.h = size[0]

    def __call__(self, img, *args, **kwargs):
        w, h = img.size
        if w/h > self.w/self.h:
            scale = w / DATA_SHAPE[1]
            new_h = int(scale * DATA_SHAPE[0])
            img = transforms.Pad((0, (new_h-h)//2))(img)
        else:
            scale = h / DATA_SHAPE[0]
            new_w = int(scale*DATA_SHAPE[1])
            img = transforms.Pad(((new_w-w)//2, 0))(img)
        img = transforms.Resize(DATA_SHAPE)(img)
        return img


DEFAULT_TRANSFORMS = transforms.Compose([
    Padding(DATA_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
