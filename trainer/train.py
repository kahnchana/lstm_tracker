import argparse
import os

import tensorflow as tf

from trainer.helpers import get_logging_tensor_hook
from trainer.model import model_fn, get_dataset
from trainer.data import kitti_data_gen, mot_data_gen, joint_data_gen


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='train_network')
    parser.add_argument('--data_path1', dest='data_path1', help='path to data JSON',
                        default="/Users/kanchana/Documents/current/FYP/fyp_2019/LSTM_Kanchana/data/kitti_tracks_{}.json")
    parser.add_argument('--data_path2', dest='data_path2', help='path to data JSON',
                        default="/Users/kanchana/Documents/current/FYP/fyp_2019/LSTM_Kanchana/data/mot_tracks_{}.json")
    parser.add_argument('--job_dir', dest='output_dir', help='model output directory',
                        default="/Users/kanchana/Documents/current/FYP/fyp_2019/LSTM_Kanchana/models/exp04")
    parser.add_argument('--lr', dest='lr', help='learning rate', default='0.001')
    parser.add_argument('--batch', dest='batch', help='batch size', default='64')
    parser.add_argument('--epochs', dest='epochs', help='num epochs', default='1000')
    parser.add_argument('--eval_int', dest='eval_int', help='eval interval in secs', default='120')
    arguments = parser.parse_args()
    return arguments


def main(_):
    args = parse_args()

    model_dir = args.output_dir
    os.makedirs(model_dir, exist_ok=True)

    # define input functions
    def train_input_fn():
        dataset = get_dataset(
            gen=joint_data_gen,
            data_path=[args.data_path1, args.data_path2],
            num_epochs=int(args.epochs),
            batch_size=int(args.batch),
            prefetch_size=int(args.batch),
            mode='train'
        )
        return dataset

    def val_input_fn():
        dataset = get_dataset(
            gen=joint_data_gen,
            data_path=[args.data_path1, args.data_path2],
            num_epochs=1,
            batch_size=int(args.batch),
            prefetch_size=int(args.batch),
            mode='val'
        )
        return dataset

    # define estimator
    run_config = tf.estimator.RunConfig(save_summary_steps=100, save_checkpoints_secs=int(args.eval_int))
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        params={
            'LEARNING_RATE': float(args.lr),
            'Eval_IOU': 0.5,
            'output_bins': 5,
            'hidden layers': [32, 32],
            'num_classes': 9,
            'num_timesteps': 10
        })

    # setup console output
    hooks = [get_logging_tensor_hook(every_n_iter=1000)]
    val_hooks = [get_logging_tensor_hook(every_n_iter=1000, tensors_to_log={'eval_precision': 'eval_precision'})]
    tf.logging.set_verbosity(tf.logging.INFO)

    # train and eval
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=val_input_fn, steps=None, throttle_secs=int(args.eval_int),
                                      hooks=val_hooks)

    tf.estimator.train_and_evaluate(estimator=classifier,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)

    # classifier.evaluate(input_fn=val_input_fn, steps=None, hooks=hooks, name='final_eval')

    return 0


if __name__ == '__main__':
    tf.app.run()
