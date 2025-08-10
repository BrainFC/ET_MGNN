import os
import util
import random
import numpy as np
from model import ModelRWKV
from dataset import *
from tqdm import tqdm
from einops import repeat
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

def step(model, criterion, dyn_v, dyn_a, sampling_endpoints, t, label, reg_lambda, clip_grad=0.0, device='cpu', optimizer=None, scheduler=None):
    # print("xxxx device xxxx",device)
    if optimizer is None: model.eval()
    else: model.train()

    # run model
    logit, attention, latent, reg_ortho = model(dyn_v.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)
    loss = criterion(logit, label.to(device))
    reg_ortho *= reg_lambda
    loss += reg_ortho

    # optimize model
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0.0: torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return logit, loss, attention, latent, reg_ortho


def train(argv):
    # make directories
    os.makedirs(os.path.join(argv.targetdir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(argv.targetdir, 'summary'), exist_ok=True)

    # set seed and device
    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)
    if torch.cuda.is_available():
        # device = torch.device("cuda")
        device = torch.device("cuda:4") # 5 报错，直接用4

        torch.cuda.manual_seed_all(argv.seed)
    else:
        device = torch.device("cpu")

    # define dataset
    if argv.dataset=='abide_246': dataset = DatasetABIDEII(argv.sourcedir, dynamic_length=argv.dynamic_length, k_fold=argv.k_fold,percent = argv.percent,bntype =argv.bntype)

    elif argv.dataset=='adni_246': dataset = DatasetADNI(argv.sourcedir, dynamic_length=argv.dynamic_length, k_fold=argv.k_fold,percent = argv.percent,bntype =argv.bntype)

    else: raise

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=argv.minibatch_size, shuffle=False, num_workers=argv.num_workers, pin_memory=True)

    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(argv.targetdir, 'checkpoint.pth')):
        # print('resuming checkpoint experiment')
        # checkpoint = torch.load(os.path.join(argv.targetdir, 'checkpoint.pth'), map_location=device)
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None}
    else:
        checkpoint = {
            'fold': 0,
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None}

    # start experiment
    for k_index, k in enumerate(dataset.folds):
        if checkpoint['fold']:
            if k_index < dataset.folds.index(checkpoint['fold']):
                continue
        # make directories per fold
        os.makedirs(os.path.join(argv.targetdir, 'model', str(k)), exist_ok=True)
        
        # set dataloader
        dataset.set_fold(k, train=True)

        # define model
        if  argv.model == "RWKV":        # test by RWKV
            model = ModelRWKV(
                input_dim=dataset.num_nodes,
                hidden_dim=argv.hidden_dim,
                num_classes=dataset.num_classes,
                num_layers=argv.num_layers,
                sparsity=argv.sparsity,
                dropout=argv.dropout,
                cls_token=argv.cls_token,
                readout=argv.readout,
            )


        model.to(device)
        model_dict = model.state_dict()
        if argv.pre_trained:
            pretrained_model = torch.load(os.path.join(argv.sourcedir, 'processed', 'model.pth'))
            filtered_dict = {k: v for k, v in pretrained_model.items() if 'linear_layers' not in k }
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
        if checkpoint['model'] is not None: model.load_state_dict(checkpoint['model'])
        criterion = torch.nn.CrossEntropyLoss() if dataset.num_classes > 1 else torch.nn.MSELoss()

        # define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=argv.max_lr, epochs=argv.num_epochs, steps_per_epoch=len(dataloader), pct_start=0.2, div_factor=argv.max_lr/argv.lr, final_div_factor=1000)
        if checkpoint['optimizer'] is not None: optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None: scheduler.load_state_dict(checkpoint['scheduler'])
        # define logging objects
        summary_writer = SummaryWriter(os.path.join(argv.targetdir,'summary', str(k), 'train'), ) #add argv.model
        summary_writer_val = SummaryWriter(os.path.join(argv.targetdir,'summary', str(k), 'val'), )
        logger = util.logger.LoggerMGNN(dataset.folds, dataset.num_classes)
        if not os.path.exists(os.path.join(argv.targetdir, argv.model, argv.bntype,'attention')):
            # os.makedirs(os.path.join(argv.targetdir, 'attention'))
            os.makedirs(os.path.join(argv.targetdir, 'attention', str(k)), exist_ok=True)

        # start training
        for epoch in range(checkpoint['epoch'], argv.num_epochs):
            logger.initialize(k)
            dataset.set_fold(k, train=True)
            loss_accumulate = 0.0
            reg_ortho_accumulate = 0.0
            # print("dataloader:",dataloader.dataset.timeseries_list[0].shape)
            for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k} e:{epoch}')):# process input data

                dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride, argv.dynamic_length)
                sampling_endpoints = [p+argv.window_size for p in sampling_points]
                if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = x['timeseries'].permute(1,0,2)
                label = x['label']

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                # print("pred:",pred," prob:",prob)
                loss_accumulate += loss.detach().cpu().numpy()
                reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], i+epoch*len(dataloader))

            # summarize results
            samples = logger.get(k)
            metrics = logger.evaluate(k)
            summary_writer.add_scalar('loss', loss_accumulate/len(dataloader), epoch)
            summary_writer.add_scalar('reg_ortho', reg_ortho_accumulate/len(dataloader), epoch)
            if dataset.num_classes > 1: summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1], epoch)
            [summary_writer.add_scalar(key, value, epoch) for key, value in metrics.items() if not key=='fold']
            
            torch.save(attention['node-attention'],os.path.join(argv.targetdir,'attention','k_' + str(k) + '_e_' + str(epoch) + '.pt'))
            [summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True), epoch) for key, value in attention.items()]
            summary_writer.flush()

            # save checkpoint
            torch.save({
                'fold': k,
                'epoch': epoch+1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
                os.path.join(argv.targetdir, 'checkpoint.pth'))

            if argv.validate:
                print('validating. not for testing purposes')
                logger.initialize(k)
                dataset.set_fold(k, train=False)
                for i, x in enumerate(dataloader):
                    with torch.no_grad():
                        # process input data
                        dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride)
                        sampling_endpoints = [p+argv.window_size for p in sampling_points]
                        if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                        if not dyn_v.shape[1]==dyn_a.shape[1]: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                        if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                        t = x['timeseries'].permute(1,0,2)
                        label = x['label']

                        logit, loss, attention, latent, reg_ortho = step(
                            model=model,
                            criterion=criterion,
                            dyn_v=dyn_v,
                            dyn_a=dyn_a,
                            sampling_endpoints=sampling_endpoints,
                            t=t,
                            label=label,
                            reg_lambda=argv.reg_lambda,
                            clip_grad=argv.clip_grad,
                            device=device,
                            optimizer=None,
                            scheduler=None,
                        )
                        pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                        prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                        logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                samples = logger.get(k)
                metrics = logger.evaluate(k)
                summary_writer_val.add_scalar('loss', loss_accumulate/len(dataloader), epoch)
                summary_writer_val.add_scalar('reg_ortho', reg_ortho_accumulate/len(dataloader), epoch)
                if dataset.num_classes > 1: summary_writer_val.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1], epoch)
                [summary_writer_val.add_scalar(key, value, epoch) for key, value in metrics.items() if not key=='fold']
                [summary_writer_val.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True), epoch) for key, value in attention.items()]
                summary_writer_val.flush()


        # finalize fold
        torch.save(model.state_dict(), os.path.join(argv.targetdir, 'model', str(k), 'model.pth'))
        checkpoint.update({'epoch': 0, 'model': None, 'optimizer': None, 'scheduler': None})

    summary_writer.close()
    summary_writer_val.close()
    os.remove(os.path.join(argv.targetdir, 'checkpoint.pth'))


def test(argv):
    os.makedirs(os.path.join(argv.targetdir, 'attention'), exist_ok=True)

    device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")

    # define dataset
    if argv.dataset == 'abide_246': dataset = DatasetABIDEII(argv.sourcedir, dynamic_length=argv.dynamic_length, k_fold=argv.k_fold,percent = argv.percent,bntype =argv.bntype)
    elif argv.dataset == 'adni_246': dataset = DatasetADNI(argv.sourcedir, dynamic_length=argv.dynamic_length,k_fold=argv.k_fold,percent = argv.percent,bntype = argv.bntype)
    else: raise
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=argv.num_workers, pin_memory=True)
    logger = util.logger.LoggerMGNN(dataset.folds, dataset.num_classes)

    save_metrics = dict()
    for k in dataset.folds:
        os.makedirs(os.path.join(argv.targetdir, 'attention', str(k)), exist_ok=True)

        # ModelRWKV
        if argv.model == "RWKV":        # test by RWKV
            model = ModelRWKV(
                input_dim=dataset.num_nodes,
                hidden_dim=argv.hidden_dim,
                num_classes=dataset.num_classes,
                num_layers=argv.num_layers,
                sparsity=argv.sparsity,
                dropout=argv.dropout,
                cls_token=argv.cls_token,
                readout=argv.readout,
            )
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(argv.targetdir, 'model', str(k), 'model.pth')))
        criterion = torch.nn.CrossEntropyLoss() if dataset.num_classes > 1 else torch.nn.MSELoss()

        # define logging objects
        fold_attention = {'node_attention': [], 'time_attention': []}
        summary_writer = SummaryWriter(os.path.join(argv.targetdir, 'summary', str(k), 'test'))

        logger.initialize(k)
        dataset.set_fold(k, train=False)
        loss_accumulate = 0.0
        reg_ortho_accumulate = 0.0
        latent_accumulate = []
        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f'k:{k}')):
            with torch.no_grad():
                # process input data
                # print("=== timeseries === shape : ",x['timeseries'].shape)
                dyn_a, sampling_points = util.bold.process_dynamic_fc(x['timeseries'], argv.window_size, argv.window_stride)
                sampling_endpoints = [p+argv.window_size for p in sampling_points]
                if i==0: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if not dyn_v.shape[1]==dyn_a.shape[1]: dyn_v = repeat(torch.eye(dataset.num_nodes), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=argv.minibatch_size)
                if len(dyn_a) < argv.minibatch_size: dyn_v = dyn_v[:len(dyn_a)]
                t = x['timeseries'].permute(1,0,2)
                label = x['label']

                logit, loss, attention, latent, reg_ortho = step(
                    model=model,
                    criterion=criterion,
                    dyn_v=dyn_v,
                    dyn_a=dyn_a,
                    sampling_endpoints=sampling_endpoints,
                    t=t,
                    label=label,
                    reg_lambda=argv.reg_lambda,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None,
                )
                pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                logger.add(k=k, pred=pred.detach().cpu().numpy(), true=label.detach().cpu().numpy(), prob=prob.detach().cpu().numpy())
                loss_accumulate += loss.detach().cpu().numpy()
                reg_ortho_accumulate += reg_ortho.detach().cpu().numpy()

                fold_attention['node_attention'].append(attention['node-attention'].detach().cpu().numpy())
                fold_attention['time_attention'].append(attention['time-attention'].detach().cpu().numpy())
                latent_accumulate.append(latent.detach().cpu().numpy())

        # summarize results
        samples = logger.get(k)
        metrics = logger.evaluate(k)
        summary_writer.add_scalar('loss', loss_accumulate/len(dataloader))
        summary_writer.add_scalar('reg_ortho', reg_ortho_accumulate/len(dataloader))
        summary_writer.add_pr_curve('precision-recall', samples['true'], samples['prob'][:,1])
        [summary_writer.add_scalar(key, value) for key, value in metrics.items() if not key=='fold']
        [summary_writer.add_image(key, make_grid(value[-1].unsqueeze(1), normalize=True, scale_each=True)) for key, value in attention.items()]
        summary_writer.flush()

        save_metrics[k] = metrics

        # finalize fold
        logger.to_csv(argv.targetdir, k)
        if 'rest' in argv.dataset:
            for key, value in fold_attention.items():
                torch.save(value, os.path.join(argv.targetdir, 'attention', str(k), f'{key}.pth'))
        elif 'task' in argv.dataset:
            for key, value in fold_attention.items():
                os.makedirs(os.path.join(argv.targetdir, 'attention', str(k), key), exist_ok=True)
                for idx, task in enumerate(dataset.task_list):
                    np.save(os.path.join(argv.targetdir, 'attention', str(k), key, f'{task}.npy'), np.concatenate([v for (v, l) in zip(value, samples['true']) if l==idx]))
        else:
            # raise
            for key, value in fold_attention.items():
                torch.save(value, os.path.join(argv.targetdir, 'attention', str(k), f'{key}.pth'))

        np.save(os.path.join(argv.targetdir, 'attention', str(k), 'latent.npy'), np.concatenate(latent_accumulate))
        del fold_attention
        del latent_accumulate

    # finalize experiment
    logger.to_csv(argv.targetdir)
    final_metrics = logger.evaluate()
    save_metrics['final'] = final_metrics
    summary_writer.close()
    print("runhere ,save test result npy")
    np.save(os.path.join(argv.targetdir,"save_metrics.npy"),save_metrics)
    torch.save(logger.get(), os.path.join(argv.targetdir, 'samples.pkl'))
