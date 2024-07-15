import argparse
from learner import Learner    

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-root_path", required=True, type=str, help="root path")
    parser.add_argument('-dataset', required=True, type=str, help="dataset")
    parser.add_argument('-conflict_ratio', required=True, type=str, help="conflict_ratio")
    parser.add_argument('-train_method', required=True, type=str, help="train method")
    parser.add_argument('-with_edited', action='store_true', help="train with edited images")
    parser.add_argument('-lr', required=True, type=float, help="learning rate")
    parser.add_argument('-epochs', required=True, type=int, help="epochs")
    parser.add_argument('-batch_size', required=True, type=int, help="batch size")
    parser.add_argument('-seed', required=True, type=int, help="random seed")
    parser.add_argument('-gpu_num', required=True, type=str, help="CUDA")
    parser.add_argument('-wandb', required=False, action='store_true', help="wandb")
    parser.add_argument('-projcode', required=True, type=str, help="project code")
    parser.add_argument('-run_name', required=False, type=str, help="run name")
    args = parser.parse_args()
    
    learner = Learner(args=args)
    learner.prepare()
    breakpoint()
    
    learner.wandb_switch(switch='start')
    
    if args.train_method == 'naive':
        # Train & eval
        for epoch in range(1, args.epochs+1):
            learner.naive_train(epoch=epoch)
            learner.eval(model_name='debiased')
            learner.wandb_log(epoch=epoch, postfix='NaiveSingleModel')
        # Save
        learner.save_model(model_name='debiased',
                           save_name='debiased.pth')
        
    elif args.train_method == 'lff':
        raise KeyError("FIXME learner lff")
    
    elif args.train_method == 'pairing':
        # Train & eval
        for epoch in range(1, args.epochs+1):
            learner.pairing_train(epoch=epoch)
            learner.eval(model_name='debiased')
            learner.wandb_log(epoch=epoch, postfix='PairingSingleModel')
        # Save
        learner.save_model(model_name='debiased',
                           save_name='debiased.pth')
        
    else:
        raise KeyError("Choose one of the two method: naive, lff")
            
    learner.wandb_switch(switch='finish')
        
        
if __name__ == '__main__':
    main()