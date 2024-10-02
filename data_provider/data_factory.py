from argparse import Namespace
from .dataset import ETThDataset, ETTmDataset, CustomDataset
from torch.utils.data import DataLoader


data_dict = {
    'ETTh1': ETThDataset,
    'ETTh2': ETThDataset,
    'ETTm1': ETTmDataset,
    'ETTm2': ETTmDataset,
    'electricity': CustomDataset,
    'exchange': CustomDataset,
    'traffic': CustomDataset,
    'weather': CustomDataset,
}

def data_provider(args: Namespace, flag: str):
    if flag == 'test':
        shuffle = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.test_batch_size
        else:
            batch_size = 1
    elif flag == 'val':
        shuffle = False
        drop_last = True
        batch_size = args.val_batch_size
    elif flag == 'train':
        shuffle = True
        drop_last = True
        batch_size = args.train_batch_size
    

    datasets = args.dataset.split('-')  # 'ETTh1-ETTh2'
    if len(datasets) == 1:
        _Dataset = data_dict[datasets[0].strip()](args=args, flag=flag)
        if args.downstream_task == 'forecasting':
            data_loader = DataLoader(
                dataset=_Dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            print(f'[{flag} dataloader] dataset: {datasets[0]}, length: {len(_Dataset)}, batch_size: {batch_size}, shuffle: {shuffle}, drop_last: {drop_last}')
        elif args.downstream_task == 'classification':
            pass
        else:
            raise ValueError(f'Invalid task: {args.downstream_task=}', 'task should be one of [forecasting, classification]')
    else:
        dataset_list = []
        for ds in datasets:
            dataset_class = data_dict[ds.strip()]
            dataset_instance = dataset_class(args=args, flag=flag, dataset=ds.strip())
            dataset_list.append(dataset_instance)
        if args.downstream_task == 'forecasting':
            dataloader_list = []
            for ds in dataset_list:
                dataloader = DataLoader(
                    dataset=ds,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )
                dataloader_list.append(dataloader)
            data_loader = dataloader_list
        elif args.downstream_task == 'classification':
            pass
        else:
            raise ValueError(f'Invalid task: {args.downstream_task=}', 'task should be one of [forecasting, classification]')

    return data_loader
