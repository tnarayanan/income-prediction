from data_input import DataInput

train_data = DataInput()
train_data.load_data()

test_data = DataInput(test_data=True)
test_data.load_data()