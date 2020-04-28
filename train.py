import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Engine, Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import (EarlyStopping, TerminateOnNan)
from ignite.metrics import (Accuracy, ConfusionMatrix, Loss, Precision, Recall,
                            RunningAverage)
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from data_utils import create_data, ordinal_to_alpha
from models import FastWeights

# Set random seed
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)


def load_data(batch_size, workers):
    X_train, y_train = create_data(64000, 3)
    X_test, y_test = create_data(32000, 3)

    # for i in range(10):
    #     print(ordinal_to_alpha([np.argmax(X) for X in X_train[i]]))
    #     print(ordinal_to_alpha([np.argmax(X) for X in X_test[i]]))
    # assert 2 == 1

    X_train, y_train = torch.as_tensor(X_train), torch.as_tensor(y_train)
    X_test, y_test = torch.as_tensor(X_test), torch.as_tensor(y_test)

    dataset_train = TensorDataset(X_train.float(), y_train.long())
    dataset_test = TensorDataset(X_test.float(), y_test.long())

    # dataset_train = TensorDataset(X_train.long(), y_train.long())
    # dataset_test = TensorDataset(X_test.long(), y_test.long())

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size,
        shuffle=True,
        num_workers=workers, pin_memory=True
        )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size,
        shuffle=False,
        num_workers=workers, pin_memory=True
        )
    return dataloader_train, dataloader_test


def run(args, writer):

    # writer = SummaryWriter(os.path.join(args['log_dir'], args['name']))

    print('Loading model....')
    training_history = {'CrossEntropy': [],
                        'Accuracy': []}
    testing_history = {'CrossEntropy': [],
                       'Accuracy': []}

    model_dict = args['config']['model']

    model = FastWeights(**model_dict)

    device = torch.device(args['config']['device'])
    model = model.to(device)

    print('Loading data....')
    train_loader, test_loader = load_data(args['config']['batch_size'], args['config']['workers'])

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params, lr=args['config']['lr'],
        momentum=args['config']['momentum'],
        weight_decay=args['config']['weight_decay']
    )

    if args['config']['scheduler'] == 'multi':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args['config']['lr_steps'],
            gamma=args['config']['lr_gamma']
        )
    elif args['config']['scheduler'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            milestones=args['config']['lr_step_size'],
            gamma=args['config']['lr_gamma']
        )
    elif args['config']['scheduler'] == 'reduce':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=args['config']['reduce_type'],
            factor=args['config']['lr_gamma'],
        )
    elif args['config']['scheduler'] == 'cyclic':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
             optimizer,
             base_lr=args['config']['lr'],
             max_lr=10*args['config']['lr']
         )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1
        )

    criterion = nn.CrossEntropyLoss()

    def evaluate_function(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = batch
            if device:
                inputs = inputs.to(device)
                targets = targets.to(device)

            inputs = torch.transpose(inputs, 0, 1)

            preds = model(inputs)
            return preds, targets

    def process_function(engine, batch):
        model.train()
        optimizer.zero_grad()
        inputs, targets = batch
        if device:
            inputs = inputs.to(device)
            targets = targets.to(device)

        inputs = torch.transpose(inputs, 0, 1)

        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        if args['config']['max_norm'] > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args['config']['max_norm']
            )
        optimizer.step()
        if args['config']['scheduler'] == 'cyclic':
            lr_scheduler.step()
        else:
            pass
        return loss.item()

    trainer = Engine(process_function)
    evaluator = Engine(evaluate_function)
    train_evaluator = Engine(evaluate_function)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    Loss(criterion, output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'CrossEntropy')
    Accuracy().attach(evaluator, 'Accuracy')

    Loss(criterion, output_transform=lambda x: [x[0], x[1]]).attach(train_evaluator, 'CrossEntropy')
    Accuracy().attach(train_evaluator, 'Accuracy')

    pbar = ProgressBar(persist=True,
                       bar_format="")
    pbar.attach(trainer, ['loss'])

    @trainer.on(Events.STARTED)
    def resume_training(engine):
        if args['config']['resume'] > 0:
            checkpoint = torch.load(os.path.join(
                args['dir'],
                args['config']['output_dir'],
                f"{args['name']}_{args['config']['resume']}.pth"
            ), map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            engine.state.epoch = args['config']['resume']

    def val_score(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['CrossEntropy']
        return -avg_loss

    def checkpointer(engine):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': engine.state.epoch,
            'args': args
        }
        save_on_master(
            checkpoint,
            os.path.join(
                args['dir'],
                args['config']['output_dir'],
                f"{args['name']}_{engine.state.epoch}.pth"
            )
        )
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              checkpointer)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    def score_function(engine):
        metrics = evaluator.state.metrics
        avg_loss = metrics['CrossEntropy']
        return -avg_loss

    def print_trainer_logs(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        avg_loss = metrics['CrossEntropy']
        avg_acc = metrics['Accuracy']*100

        training_history['CrossEntropy'].append(avg_loss)
        training_history['Accuracy'].append(avg_acc)

        writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_acc, engine.state.epoch)

        print(
            "Training Results - Epoch: {} ".format(engine.state.epoch),
            "Avg loss: {:.4f} ".format(avg_loss),
            "Avg Acc: {:.4f} ".format(avg_acc)
        )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, print_trainer_logs)

    def log_validation_results(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['CrossEntropy']
        avg_acc = metrics['Accuracy']*100

        if args['config']['scheduler'] == 'reduce':
            lr_scheduler.step(-avg_loss)
        elif args['config']['scheduler'] == 'cyclic':
            pass
        else:
            lr_scheduler.step()

        testing_history['CrossEntropy'].append(avg_loss)
        testing_history['Accuracy'].append(avg_acc)

        writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar("validation/avg_accuracy", avg_acc, engine.state.epoch)

        print(
            "Validation Results - Epoch: {} ".format(engine.state.epoch),
            "Avg loss: {:.4f} ".format(avg_loss),
            "Avg Acc: {:.4f} ".format(avg_acc)
        )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

    handler = EarlyStopping(patience=args['config']['patience'],
                            score_function=score_function,
                            trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    print('Training....')
    trainer.run(train_loader, max_epochs=args['config']['epochs'])
    writer.close()
    np.save(os.path.join(args['dir'], f"{args['name']}_traininglog.npy"), [training_history])
    np.save(os.path.join(args['dir'], f"{args['name']}_testinglog.npy"), [testing_history])
