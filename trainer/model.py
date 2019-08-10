import tensorflow as tf
from trainer.helpers import bbox_overlap_iou, to_bbox_tf

Layers = tf.keras.layers


def build_lstm(hidden_sizes, reuse=False):
    """
    Builds the LSTM network
    Args:
        hidden_sizes:
        reuse:

    Returns:
        tf.keras.Layers.RNN instance (stacked LSTM cells)
    """
    with tf.variable_scope("LSTM") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        cells = []
        for i, hidden_size in enumerate(hidden_sizes):
            cell = Layers.LSTMCell(units=hidden_size, unit_forget_bias=1, name="LSTM_Layer_{}".format(i))
            cells.append(cell)
        outputs = Layers.RNN(cells, name="RNN_net")  # RNN cell composed sequentially of multiple simple cells.

        tf.get_variable_scope().reuse_variables()

        return outputs


def create_model(timesteps=None, input_dim=4, num_classes=9, hidden_sizes=(32, 32), num_outputs=5):
    """
    Creates a Keras Model with the LSTM
    Args:
        timesteps:
        input_dim:
        num_classes:
        hidden_sizes:
        num_outputs:

    Returns:
        tf.keras.Model
    """
    inputs = tf.keras.Input(shape=(timesteps, input_dim + num_classes), name="input_tensor")
    # LSTM component of network built from function above
    lstm = build_lstm(hidden_sizes=hidden_sizes)(inputs)

    x_conf = Layers.Dense(units=num_outputs, name='x_conf')(lstm)
    x_reg = Layers.Dense(units=num_outputs, name='x_reg')(lstm)
    x_comb = Layers.concatenate(inputs=[x_conf, x_reg], axis=-1, name='x_comb')

    y_conf = Layers.Dense(units=num_outputs, name='y_conf')(lstm)
    y_reg = Layers.Dense(units=num_outputs, name='y_reg')(lstm)
    y_comb = Layers.concatenate(inputs=[y_conf, y_reg], axis=-1, name='y_comb')

    h_conf = Layers.Dense(units=num_outputs, name='h_conf')(lstm)
    h_reg = Layers.Dense(units=num_outputs, name='h_reg')(lstm)
    h_comb = Layers.concatenate(inputs=[h_conf, h_reg], axis=-1, name='h_comb')

    w_conf = Layers.Dense(units=num_outputs, name='w_conf')(lstm)
    w_reg = Layers.Dense(units=num_outputs, name='w_reg')(lstm)
    w_comb = Layers.concatenate(inputs=[w_conf, w_reg], axis=-1, name='w_comb')

    outputs = [x_comb, y_comb, h_comb, w_comb]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_model")

    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(lr=0.001),
    #     loss=tf.keras.losses.mean_squared_error,
    #     metrics=[tf.keras.metrics.mean_squared_error]
    # )

    return model


def model_fn(features, labels, mode, params, config):
    """
    Args:
        features:   This is the first item returned from the `input_fn` passed to `train`, `evaluate`, and
                    `predict`. This should be a single `tf.Tensor` or `dict` of same.
        labels:     This is the second item returned from the `input_fn` passed to `train`, `evaluate`, and
                    `predict`. This should be a single `tf.Tensor` or `dict` of same (for multi-head models).
                    If mode is `tf.estimator.ModeKeys.PREDICT`, `labels=None` will be passed. If the `model_fn`'s
                    signature does not accept `mode`, the `model_fn` must still be able to handle `labels=None`.
        mode:       Optional. Specifies if this training, evaluation or prediction. See `tf.estimator.ModeKeys`.
        params:     Optional `dict` of hyperparameters.  Will receive what is passed to Estimator in `params`
                    parameter. This allows to configure Estimators from hyper parameter tuning.
        config:     Optional `estimator.RunConfig` object. Will receive what is passed to Estimator as its `config`
                    parameter, or a default value. Allows setting up things in your `model_fn` based on
                    configuration such as `num_ps_replicas`, or `model_dir`.

    Returns:
        tf.estimator.EstimatorSpec

    """
    # load model
    num_outputs = 5
    model = create_model(timesteps=10, input_dim=4, num_classes=9, hidden_sizes=(32, 32), num_outputs=num_outputs)
    # tf.keras.utils.plot_model(model, "/Users/kanchana/Desktop/model.png")

    # prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = model(features, training=False)
        predictions = tf.stack(predictions, axis=-2)
        predictions = to_bbox_tf(x=features, y=predictions, batch=True)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'boxes': tf.estimator.export.PredictOutput(predictions)
            })

    # train mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        predictions = model(features)
        predictions = tf.stack(predictions, axis=-2)

        # define loss function
        idx = tf.argmax(labels[:, :, :num_outputs], axis=-1, output_type=tf.int32)
        dist = predictions[:, :, num_outputs:] * tf.one_hot(idx, num_outputs)
        dist_loss = tf.losses.huber_loss(labels=labels[:, :, num_outputs:], predictions=dist)
        conf_loss = 1e3 * tf.losses.softmax_cross_entropy(onehot_labels=labels[:, :, :num_outputs],
                                                          logits=predictions[:, :, :num_outputs])
        loss = conf_loss + dist_loss

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=params["LEARNING_RATE"])

        # define metrics
        predictions = to_bbox_tf(x=features, y=predictions, batch=True)
        labels = to_bbox_tf(x=features, y=labels, batch=True)
        accuracy = tf.metrics.mean_squared_error(predictions, labels)

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(params["LEARNING_RATE"], 'learning_rate')
        tf.identity(loss, 'mean_squared_loss')
        tf.identity(accuracy[1], name='train_accuracy')

        tf.summary.scalar('train_accuracy', accuracy[1])
        tf.summary.scalar('conf_loss', conf_loss)
        tf.summary.scalar('dist_loss', dist_loss)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()),
        )

    # eval mode
    if mode == tf.estimator.ModeKeys.EVAL:
        predictions = model(features, training=False)
        predictions = tf.stack(predictions, axis=-2)

        loss = tf.losses.huber_loss(labels=labels, predictions=predictions)

        predictions = to_bbox_tf(x=features, y=predictions, batch=True)
        labels = to_bbox_tf(x=features, y=labels, batch=True)

        # metrics
        accuracy = tf.metrics.mean_squared_error(predictions, labels)
        # raw iou values
        iou_val = bbox_overlap_iou(predictions[:, :4], labels[:, :4], ar=False)
        m_iou = tf.metrics.mean(values=iou_val, name="mean_iou")
        # iou as bool
        iou = tf.cast(tf.greater_equal(iou_val, params["Eval_IOU"]), tf.int8)
        precision = tf.metrics.accuracy(predictions=iou, labels=tf.ones_like(iou))

        tf.identity(precision[1], name='eval_precision')
        tf.summary.scalar('eval_precision', precision[1])
        tf.summary.scalar('eval_mean_iou', m_iou[1])
        tf.summary.scalar('eval_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'eval_precision': precision,
                'eval_accuracy': accuracy,
                'eval_mean_iou': m_iou
            }
        )


def get_dataset(gen=None,
                data_path="/Users/kanchana/Documents/current/FYP/data/KITTI_tracking/generate/tracks.json",
                mode="train",
                num_classes=9,
                num_epochs=100,
                batch_size=10,
                prefetch_size=10):
    # def gen():
    #     temp = kitti_data_gen(path=data_path, split=mode)
    #     yield next(temp)

    dat = tf.data.Dataset.from_generator(generator=gen,
                                         output_types=(tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape((10, 4 + num_classes)), tf.TensorShape((4, 10))),
                                         args=(data_path, mode))
    dat = dat.repeat(count=num_epochs).batch(batch_size=batch_size).prefetch(buffer_size=prefetch_size)

    return dat
