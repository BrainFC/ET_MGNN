import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description='efficient temporal multi-modal graph neural network')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--exp_name', type=str, default='experiment_et_mgnn')
    parser.add_argument('-k', '--k_fold', default=5) #  strde =5时， batchsize继续从12减小到6
    parser.add_argument('-b', '--minibatch_size', type=int, default =24) #64，16，原始为1,从16改为8；GIN：24, GIN可以用32
    parser.add_argument('-p', '--pre_trained', type=bool, default= False)
    parser.add_argument('-ds', '--sourcedir', type=str, default='/home/image015/BrainCode/data/')
    parser.add_argument('--device',type=str,default='cuda:0',help='device')

    parser.add_argument('-dt', '--targetdir', type=str,default='/home/image015/BrainCode/Disease/result_Ablation/')

    parser.add_argument('--dataset', type=str, default='abide_246', choices=['ppmi_246','abide_246','adni_246'])

    parser.add_argument('--fwhm', type=float, default=None)
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--ds', type=int, default=2)
    parser.add_argument('--window_stride', type=int, default=10)
    parser.add_argument('--dynamic_length', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--reg_lambda', type=float, default=0.0001)
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default = 50)
    parser.add_argument('--num_heads', type=int, default= 1)
    parser.add_argument('--num_layers', type=int, default= 4)
    parser.add_argument('--hidden_dim', type=int, default= 64)
    parser.add_argument('--sparsity', type=int, default= 30)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--readout', type=str, default='garo', choices=['garo', 'sero', 'mean','baro'])
    parser.add_argument('--cls_token', type=str, default='sum', choices=['sum', 'mean', 'param'])

    parser.add_argument('--num_clusters', type=int, default=4) #原始为7
    parser.add_argument('--subsample', type=int, default=50)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--validate', action='store_true')

    parser.add_argument('--num_workers', type=int, default=32)#16改为12
    parser.add_argument('--num_samples', type=int, default=-1)

    # SC (0-1), SC2( 1 ~ 無窮大）
    parser.add_argument('--bntype', type=str, default='FC',choices=['FC','SC'])

    parser.add_argument('--percent', type=int, default=30) # 75,85,90,85
    parser.add_argument('--percent_sc', type=int, default=50) # 75,85,90

    parser.add_argument('--th1', type=float, default=0.4) # 75,85,90
    #AN : AD vs. NC , AM: AD vs. MCI,   NM: NC vs. MCI
    parser.add_argument('--cltype', type=str, default='NM',choices=['AN','AM','NM'])

    parser.add_argument('--model', type=str, default='RWKV',choices=['RWKV'])

    argv = parser.parse_args()

    argv.targetdir = os.path.join(argv.targetdir,argv.dataset, argv.model, '_window_'
                                  +str(argv.window_size)+ '_'+str(argv.sparsity),
                                  argv.bntype + '_' + argv.readout + '_' + argv.cltype +
                                  '_sparsity_'+str(argv.sparsity)+ str(argv.percent) + '_percent_sc_' +
                                  str(argv.percent_sc) + '_layers' + str(argv.num_layers) + '_stride_' +
                                  str(argv.window_stride) + '_dynamic_length_' + str(argv.dynamic_length))
    print(argv.targetdir)
    os.makedirs(argv.targetdir, exist_ok=True)
    with open(os.path.join(argv.targetdir,'argv.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(argv).items())
    return argv

