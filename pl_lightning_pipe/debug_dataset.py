from pathlib import Path
from clearml import Dataset, Task
from pipe_conf import PROJECT_NAME



if __name__ == '__main__':
    
    dataset_path = Dataset.get(
            dataset_project=PROJECT_NAME,
            dataset_name='data_split'
    ).get_local_copy()


    print(dataset_path)
    train = list(Path(dataset_path).glob('train.json'))[0]
    print(train)
        # print(str(f).split('.')[-2])
        # if str(f).split('.')[-1] == 'train':
        #     train = f
        # elif str(f).split('.')[-1] == 'test':
        #     test = f
        # elif str(f).split('.')[-1] == 'validation':
        #     validation = f
    


    