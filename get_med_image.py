from medmnist import INFO, Evaluator
import medmnist
import os
from tqdm import tqdm
data_flags = ["breastmnist","bloodmnist","organamnist","organsmnist","organcmnist","pneumoniamnist","dermamnist","tissuemnist","octmnist","PathMNIST","chestmnist"]
for data_flag in data_flags:
    data_flag = data_flag.lower()

    info = INFO[data_flag]

    data_root = "/root/autodl-tmp/"
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if 'binary-class' not in info["task"] and 'multi-class' not in info["task"]:
        print(f'task:{info["task"]}')
        raise "Unsupported dataset task"
    DataClass = getattr(medmnist, info['python_class'])

    # validation_dataset = DataClass(split='val', as_rgb=True,download=True, size=224, transform=TwoCropTransform(trans['valid']))
    # test_dataset = DataClass(split='test', as_rgb=True, download=True, size=224, transform=TwoCropTransform(trans['test']))
    # training_dataset = DataClass(split='train', as_rgb=True, download=True,size=224,transform=TwoCropTransform(trans['train']))
    # validation_dataset = DataClass(split='val', root=data_root, download=True, transform=None,
    #                                as_rgb=True, size=224)
    # test_dataset = DataClass(split='test', root=data_root, download=True, transform=None,as_rgb=True, size=224)
    # training_dataset = DataClass(split='train', root=data_root, download=True, transform=None,
    #                              as_rgb=True, size=224)
    label_to_str = info['label']
    name = info['python_class']
    description = info['description']
    n_channels = info['n_channels']
    n_samples = info['n_samples']
    print(f'{name}|{len(label_to_str)}|{n_samples["train"]}|{n_samples["val"]}|{n_samples["test"]}|224*224|Public|{n_channels}|{description}')

    # for i,(image,label) in enumerate(tqdm(test_dataset)):
    #
    #     label = str(label[0])
    #     label = label_to_str[label]
    #     save_path = os.path.join(data_root, "med_test", data_flag, label)
    #     os.makedirs(save_path, exist_ok=True)
    #     image.save(os.path.join(save_path,str(i)+".png"))
