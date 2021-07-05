import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num1', type=int, default=1e-8, help='first number')
    parser.add_argument('--num2', type=int, default=1e-8, help='second number')
    parser.add_argument('--operation', type=str, default=None, help='operation')
    args = parser.parse_args()

    print(args.num1)
    print(args.num2)
    
