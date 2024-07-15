from learner import Learner    
import argparse
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', required=True, type=str, help="model_type")
    parser.add_argument('-pretrained', required=False, action='store_true', help="pretrained")
    parser.add_argument('-dataset', required=True, type=str, help="dataset")
    parser.add_argument('-lr', required=True, type=float, help="learning rate")
    parser.add_argument('-epochs', required=True, type=int, help="epochs")
    parser.add_argument('-batch_size', required=True, type=int, help="batch size")
    parser.add_argument('-conflict_ratio', required=True, type=float, help="conflict_ratio")
    parser.add_argument('-seed', required=True, type=int, help="random seed")
    parser.add_argument('-projcode', required=True, type=str, help="project code")
    parser.add_argument('-wandb', required=False, action='store_true', help="wandb")
    parser.add_argument('-run_name', required=False, type=str, help="run name")
    parser.add_argument('-gpu_num', required=True, type=str, help="CUDA")
    args = parser.parse_args()
    
    learner = Learner(args)
    learner.prepare()
    learner.wandb_switch(switch='start', run_name=args.run_name)
    
    for epoch in range(args.epochs):
        learner.train(epoch=epoch)
        # Eval biased model
        learner.eval(model_name='biased')
        learner.wandb_log(epoch=epoch, postfix='biased')
        
        # Eval debiased model
        learner.eval(model_name='debiased')
        learner.wandb_log(epoch=epoch, postfix='debiased')
        
    learner.wandb_switch(switch='finish')
        
    learner.save_model('biased', 'biased995.pth')
    learner.save_model('debiased', 'debiased995.pth')
        

if __name__ == '__main__':
    main()