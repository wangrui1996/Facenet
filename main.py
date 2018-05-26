import sys
from src.gress import progress
from src.gress import GetFace
import argparse

def main(args):
    #cam = args.cap
    port = args.port
    mode = args.mode
    record = args.record
    if mode in {0, 1, 2}:
        progress(port, mode, record)   # train or classifier
    else:
        GetFace()              # get face from Video or translate face of images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, help='Indicates the mode, 0: Euclidian Classification, 1: SVM Classification, 2: Train SVM, 3: Get Face', default=0)
#    parser.add_argument('cap', type=str, help='Indicates if a path of video or via of camera, default is the number of camera zero', default='0')
    parser.add_argument('--port', type=int, help='Indicates the number of port is use by TCP, default is the number of 9999', default=9999)
    parser.add_argument('--record', type=bool, help="Indicates if to record", default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

