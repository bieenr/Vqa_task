import json
def load_data(file):
    train_data = json.load(open(file,'r'))
    print(len(train_data['data']))
    print(train_data['data'][0])
if __name__ == '__main__':
    load_data(r'vqa_train.json')