import os
from stemgnn.model import StemGNN
from stemgnn.utils.data.forecast_dataset import ForecastDataModule
import argparse
import pandas as pd
import pytorch_lightning as pl


parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='ECG_data')
parser.add_argument('--window_size', type=int, default=50)
parser.add_argument('--horizon', type=int, default=15)
parser.add_argument('--train_length', type=float, default=7)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=1)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--norm_method', type=str, default='min_max')
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.2)
parser.add_argument('--dropout_rate', type=float, default=0.2)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)



if __name__ == '__main__':

    args = parser.parse_args()
    print(f'Training configs: {args}')
    data_file = os.path.join('dataset', args.dataset + '.csv')
    data = pd.read_csv(data_file).values

    data_module = ForecastDataModule(
        data, 
        batch_size=args.batch_size, 
        window_size=args.window_size,
        horizon=args.horizon,
    )

    model = StemGNN(
        node_count=data.shape[1],
        stack_cnt=2,
        time_step=args.window_size,
        multi_layer=args.multi_layer,
        horizon=args.horizon,
        dropout_rate=args.dropout_rate,
        leaky_rate=args.leakyrelu_rate,
        learning_rate=args.lr,
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val/mse',
        patience=5,
        verbose=True,
        mode='min',
        min_delta=0.001,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epoch,
        detect_anomaly=True,
        callbacks=[early_stopping_callback],
    )

    trainer.fit(model, data_module)

