import os
import numpy as np
import pandas as pd
import math

import torch
from torchvision import datasets, models, transforms
from prediction.LotoNet import LotoNet
from prediction.datasets import LotteryDataset

# python .\interfaces\predictNumbers.py --lottery_type loto7 --checkpoint_path 'F:/repositories/workspaces/prediction_num/prediction/checkpoint/checkpoint_epoch(500)_loto7_2021_02_25_1521.pth'

# parser = argparse.ArgumentParser()
# parser.add_argument("--lottery_type", type=str, default='loto7', help="lottery type to predict")
# opt = parser.parse_args()

class PredictNumbers():

    def __init__(self):

        path = './prediction/checkpoint/'
        self.ckp_file = {
            'loto7': os.path.join(path, 'checkpoint_epoch(500)_loto7_2021_02_25_1521.pth'),
            'loto6': os.path.join(path, 'checkpoint_epoch(500)_loto6_2021_03_09_1150.pth'),
            'miniloto': os.path.join(path, 'checkpoint_epoch(200)_miniloto_2021_03_09_1303.pth'),
            'numbers3': os.path.join(path, 'checkpoint_epoch(100)_numbers3_2021_02_25_1642.pth'),
            'numbers4': os.path.join(path, 'checkpoint_epoch(100)_numbers4_2021_03_09_1513.pth')}
        self.model = {}
        
        self.max_numbers = {
            'loto7': 7,
            'loto6': 6,
            'miniloto': 5,
            'numbers3': 3,
            'numbers4': 4}

        self.lottery_range = {'loto7': (1, 38), 'loto6': (1, 44), 'miniloto': (1, 32), 'numbers3': (0, 10), 'numbers4': (0, 10)}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for key in self.ckp_file:
            checkpoint = torch.load(self.ckp_file[key], map_location=device)
            model = LotoNet()
            model.eval()
            model.load_state_dict(checkpoint)
            self.model[key] = model

    def predict(self, data, lottery_type): 

        df = pd.DataFrame()
        for item in data:
            df = df.append(pd.DataFrame(item, index=[0]))
        df = df.drop(['id'], axis=1)

        t = transforms.Compose([])
        dataset_tools = LotteryDataset(root=None, lottery_type=lottery_type, transform=t)

        data_bin = dataset_tools.dataframe_to_bin(df)
        data_bin = np.expand_dims(np.expand_dims(data_bin, axis=0), axis=0)
        with torch.no_grad():
            y_hat = self.model[lottery_type](torch.from_numpy(data_bin))

        pred_num = dataset_tools.bin_to_numbers(y_hat.cpu()) + 1
        return pred_num[0].tolist()

    def statistics_predict(self, data, lottery_type): 

        l_max = self.max_numbers[lottery_type]

        df = pd.DataFrame()
        for item in data:
            df = df.append(pd.DataFrame(item, index=[0]))
        df = df.drop(['id'], axis=1)
        r = df.iloc[:, 2: 2 + l_max].to_numpy()

        lottery_data = np.arange(*self.lottery_range[lottery_type])
        lottery_length = len(lottery_data)

        rf = r.flatten()

        point_count_std = len(rf) // lottery_length

        point_even_odd_std = np.count_nonzero((lottery_data % 2) == 1) / np.count_nonzero((lottery_data % 2) == 0) 
        point_odd_std = np.count_nonzero((r % 2) == 1)
        point_even_std = np.count_nonzero((r % 2) == 0)
        point_even_odd_std = math.ceil(100 * (point_even_odd_std - point_odd_std / point_even_std))

        point_avg_std = np.average(rf)

        a = np.ceil(np.average(r[: 10]) * np.array([0.618, 0.382, 0.236])) + 1
        b = np.ceil(np.average(r[: 10]) * np.array([0.618, 0.382, 0.236])) - 1
        point_fibonacci_std = np.concatenate((a, b))

        result = []
        for i in lottery_data:

            # 出现概率
            point_count = point_count_std - np.count_nonzero(rf == i)

            if i % 2 == 1:
                # 奇数率
                point_even_odd = point_even_odd_std
            else:
                # 偶数率
                point_even_odd = 0

            # 大数概率
            point_avg = 0 if i > point_avg_std else 2

            # 斐波那契数列
            point_fibonacci = 5 * np.count_nonzero(point_fibonacci_std == i)

            print(i, point_count, point_even_odd, point_avg, point_fibonacci, sep=',\t')

            result.append(point_count + point_even_odd + point_avg + point_fibonacci)

        return np.argpartition(-np.array(result), l_max)[: l_max].tolist()

        # return pred_num[0].tolist()

# if __name__ == '__main__':

#     import sys
#     sys.path.append("interfaces")

#     import postgresDB as postgresDB
#     from LotteryData import LotteryData

#     with open('F:/repositories/workspaces/prediction_num/app_dev.yaml', 'r') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#         config = edict(config)

#     os.environ['DB_HOST'] = config.env_variables.DB_HOST
#     os.environ['DB_USER'] = config.env_variables.DB_USER
#     os.environ['DB_PASS'] = config.env_variables.DB_PASS
#     os.environ['DB_NAME'] = config.env_variables.DB_NAME

#     db = postgresDB.init_connection_engine()
#     lottery = LotteryData(db)

#     lottery_data = lottery.get_lottery_list(lottery_type='loto7', offset=0, limit=64)
#     # checkpoint_path = 'F:/repositories/workspaces/prediction_num/prediction/checkpoint/checkpoint_epoch(500)_loto7_2021_02_25_1521.pth'

#     p = PredictNumbers()

#     result = p.predict(lottery_data, opt.lottery_type)
#     print(list(result))
#     print(result)