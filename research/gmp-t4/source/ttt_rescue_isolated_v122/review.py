import sys,runpy,argparse
from pathlib import Path
R=Path('/public/home/2023141520023/return_selector_20261006');sys.path.insert(1,str(R/'ttt_rescue_train_v118'))
import audit_results as A
A.FITS=R/'ttt_rescue_fit_v122_output';A.CONF=R/'ttt_rescue_confirm_v123_output';A.OUT=R/'ttt_rescue_review_v124_output'
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation');args=parser.parse_args()
    if args.operation=='audit':A.main()
    else:runpy.run_path(str(R/'ttt_rescue_train_v118'/f'{args.operation}.py'),run_name='__main__')
