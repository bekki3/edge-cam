from .voc_detection import PASCAL_VOC_Dataset, Merging_Dataset

def load_dataset(args, transforms=None, eval_mode_value=0):
    class_path = args.class_path
    datasets = []
    print("Dataset parts: ", len(args.dataset_root.split(','))*len(args.dataset_domains.split(',')))
    for data_dir in args.dataset_root.split(','):
        for domain in args.dataset_domains.split(','):
            datasets.append(PASCAL_VOC_Dataset(root_dir=data_dir,
                                               domain=domain,
                                               transform=transforms,
                                               class_path=class_path,
                                               eval_mode=eval_mode_value))
    merged_dataset = Merging_Dataset(datasets)
    return merged_dataset
