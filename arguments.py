import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('--data_path', type=str, help='Path to the data',
                        default='data')
    parser.add_argument('--ehr_path', type=str, help='Path to the ehr data',
                        default='data/ehr')
    parser.add_argument('--cxr_path', type=str, help='Path to the cxr data',
                        default='data/cxr')

    parser.add_argument('--timestep', type=float, default=1.0, help="fixed timestep used in the dataset")
    parser.add_argument('--normalizer_state', type=str, default=None, help='Path to a state file of a normalizer. Leave none if you want to use one of the provided ones.')
    parser.add_argument('--resize', default=256, type=int, help='number of epochs to train')
    parser.add_argument('--crop', default=224, type=int, help='number of epochs to train')

    parser.add_argument('--task', type=str, default='readmission', help='in-hospital-mortality,length-of-stay,decompensation,phenotyping,readmission')
    parser.add_argument('--epochs', type=int, default=10, help='number of chunks to train')
    parser.add_argument('--device', type=str, default="cpu", help='cuda or cpu')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers for dataloader')
    parser.add_argument('--seed', type=int, default=40, help='random seed: 40,42,44...')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--model', type=str, default='our', help='our model(SaCal)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)

    return parser
