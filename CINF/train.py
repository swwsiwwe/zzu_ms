import mindspore.nn as nn
from mindspore import ops
import mindspore
from mindspore import ms_function
import ml_collections
from get_dataset import *
from metric import *
from model.blackres_ms import JAFFNet
from mindspore import context

mindspore.set_seed(1)

def get_config():
    """configuration """
    config = ml_collections.ConfigDict()
    config.epochs = 300
    config.train_data_path = "./datasets/dagm/train"
    config.val_data_path = "./datasets/dagm/val"
    config.imgsize = 224
    config.batch_size = 8
    config.pretrained_path = None
    config.in_channel = 3
    config.n_classes = 1
    #config.lr = 0.1
    return config


cfg = get_config()
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
train_dataset = create_dataset(cfg.train_data_path, img_size=cfg.imgsize, batch_size=cfg.batch_size, augment=True,
                               shuffle=True)
val_dataset = create_dataset(cfg.val_data_path, img_size=cfg.imgsize, batch_size=cfg.batch_size, augment=False,
                             shuffle=False)
net = JAFFNet()

criterion = nn.BCEWithLogitsLoss()


milestone = [2, 5, 10, 30, 100, 300]
learning_rates = [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]
lr = nn.dynamic_lr.piecewise_constant_lr(milestone, learning_rates)
optimizer = nn.AdamWeightDecay(params=net.trainable_params(),learning_rate=lr)
#optimizer = nn.SGD(params=net.trainable_params(), learning_rate=cfg.lr)
#optimizer = nn.Momentum(params=net.trainable_params(), learning_rate=cfg.lr,momentum=0.9)
# exponential_decay_lr = nn.ExponentialDecayLR(0.1, 0.9, 4)
# optimizer=nn.Momentum(net.trainable_params(), learning_rate=exponential_decay_lr, momentum=0.9)

iters_per_epoch = train_dataset.get_dataset_size()
total_train_steps = iters_per_epoch * cfg.epochs
print('iters_per_epoch: ', iters_per_epoch)
print('total_train_steps: ', total_train_steps)

metrics_name = ["acc", "iou", "dice", "sens", "spec"]
best_iou = 0
ckpt_path = 'checkpoint/CINF.ckpt'


def train(model, dataset, loss_fn, optimizer, met):
    # Define forward function
    def forward_fn(data, label):
        logits1,logits2,logits3,logits4,logits5 = model(data)
        loss1 = loss_fn(logits1, label)
        loss2 = loss_fn(logits2, label)
        loss3 = loss_fn(logits3, label)
        loss4 = loss_fn(logits4, label)
        loss5 = loss_fn(logits5, label)
        return (loss1, loss2,loss3,loss4,loss5), (logits1, logits2,logits3,logits4,logits5)

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    @ms_function
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss, logits

    size = dataset.get_dataset_size()
    model.set_train(True)
    train_loss = 0
    train_pred = []
    train_label = []
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss, logits = train_step(data, label)
        train_loss += loss[0].asnumpy()
        train_loss += loss[1].asnumpy()
        train_loss += loss[2].asnumpy()
        train_loss += loss[3].asnumpy()
        train_loss += loss[4].asnumpy()
        train_pred.extend(logits[0].asnumpy())
        train_label.extend(label.asnumpy())

    train_loss /= size
    metric = metrics_(met, smooth=1e-5)
    metric.clear()
    metric.update(train_pred, train_label)
    res = metric.eval()
    print(f'Train loss:{train_loss:>4f}', '丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (
    res[0], res[1], res[2], res[3], res[4]))


def val(model, dataset, loss_fn, met):
    size = dataset.get_dataset_size()
    model.set_train(False)
    val_loss = 0
    val_pred = []
    val_label = []
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        pred1,pred2,pred3,pred4,pred5 = model(data)
        val_loss += loss_fn(pred1, label).asnumpy()
        val_loss += loss_fn(pred2, label).asnumpy()
        val_loss += loss_fn(pred3, label).asnumpy()
        val_loss += loss_fn(pred4, label).asnumpy()
        val_loss += loss_fn(pred5, label).asnumpy()
        val_pred.extend(pred1.asnumpy())
        val_label.extend(label.asnumpy())

    val_loss /= size
    metric = metrics_(met, smooth=1e-5)
    metric.clear()
    metric.update(val_pred, val_label)
    res = metric.eval()

    print(f'Val loss:{val_loss:>4f}', '丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (
        res[0], res[1], res[2], res[3], res[4]))

    checkpoint = res[1]
    return checkpoint, res[4]


for epoch in range(cfg.epochs):
    print(f"Epoch [{epoch + 1} / {cfg.epochs}]")
    train(net, train_dataset, criterion, optimizer, metrics_name)
    checkpoint_best, spec = val(net, val_dataset, criterion, metrics_name)
    if epoch > 2 and spec > 0.2:
        if checkpoint_best > best_iou:
            print('IoU improved from %0.4f to %0.4f' % (best_iou, checkpoint_best))
            best_iou = checkpoint_best
            mindspore.save_checkpoint(net, ckpt_path)
            print("saving best checkpoint at: {} ".format(ckpt_path))
        else:
            print('IoU did not improve from %0.4f' % (best_iou), "\n-------------------------------")

