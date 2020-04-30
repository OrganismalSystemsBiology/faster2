import argparse
import os
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target_dir", required=True, help="path to the target directory")
    parser.add_argument("-o", "--output_dir", required=True, help="path to the directory of the resulting video clips")
    
    args = parser.parse_args()

    target_dir = args.target_dir
    output_dir = args.output_dir

    file_list = glob(os.path.join(target_dir, '**/*.avi'))
    

    print('list')
    print('\n'.join(file_list))
