from review import A,R,runpy,argparse
from statistics_v124 import summarize
A.summarize=summarize
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation');args=parser.parse_args()
    if args.operation=='audit':A.main()
    else:runpy.run_path(str(R/'ttt_rescue_train_v118'/f'{args.operation}.py'),run_name='__main__')
