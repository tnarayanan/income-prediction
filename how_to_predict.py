import predict

files = [
    '14_2794_6539.jpg',
    '14_2801_6536.jpg',
    '14_2802_6548.jpg'
]

for f in files:
    print(predict.predict('data/imagery/' + f))