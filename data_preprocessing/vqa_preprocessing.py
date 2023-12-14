import json 
def processing():
    train = []
    test = []
    imdir='abstract_v002_%s_%012d.png'
    print ('Loading annotations and questions...')
    train_anno = json.load(open(r'data_raw/Annotations_Binary_Train2017_abstract_v002/abstract_v002_train2017_annotations.json', 'r'))
    val_anno = json.load(open(r'data_raw/Annotations_Binary_Val2017_abstract_v002/abstract_v002_val2017_annotations.json', 'r'))

    train_ques = json.load(open(r'data_raw/Questions_Binary_Train2017_abstract_v002/OpenEnded_abstract_v002_train2017_questions.json', 'r'))
    val_ques = json.load(open(r'data_raw/Questions_Binary_Val2017_abstract_v002/OpenEnded_abstract_v002_val2017_questions.json', 'r'))
    # print(train_anno.keys())

    subtype = 'train2015'
    for i in range(len(train_anno['annotations'])):
        ans = train_anno['annotations'][i]['multiple_choice_answer']
        question_id = train_anno['annotations'][i]['question_id']
        question = train_ques['questions'][i]['question']
        image_path = imdir%(subtype, train_anno['annotations'][i]['image_id'])
        # print(train_anno['annotations'][i].keys())
        train.append({'question_id': question_id, 'image_path': image_path, 'question': question, 'answer': ans})

    subtype = 'val2015'
    for i in range(len(val_anno['annotations'])):
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        question_id = val_anno['annotations'][i]['question_id']
        # print(val_ques.keys())
        question = val_ques['questions'][i]['question']
        image_path = imdir%(subtype, val_anno['annotations'][i]['image_id'])
        test.append({'question_id': question_id, 'image_path': image_path, 'question': question, 'answer': ans})
    print('Training sample %d, Testing sample %d...' %(len(train), len(test)))
    json.dump({'data':train}, open('vqa_train.json', 'w'))
    json.dump({'data':test}, open('vqa_test.json', 'w'))
    return

if __name__ == "__main__":
    processing()