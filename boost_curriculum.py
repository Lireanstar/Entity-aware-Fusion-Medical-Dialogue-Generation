
train_data = torch.load("../data_processed/train_data_all_clean.pth")  ## whole train
'''
(encoder_input, decoder_input, encoder_target,
                               attribute_label, predict_topic_label, predict_item_label)
'''
all_train_data = [[train_data[0][i]] + [train_data[1][i]] + [train_data[2][i]] + \
                  [train_data[3][i]] + [train_data[4][i]] + [train_data[5][i]] \
                  for i in range(len(train_data[0]))]
del train_data
all_data = pd.DataFrame(all_train_data)

# For fold results
# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
# ------------------------END CROSS VALIFICATION SETTING--------------

# ------------------------START TRAINING-------------------
update_count = 0
count_ = 0
start = time.time()
print('start training....')
for train_index, val_index in kfold.split(range(len(all_train_data))):
    print("K fold is {}".format(count_))
    train = all_data.loc[train_index, :].reset_index()
    train_ = train.to_numpy().tolist()
    train_1 = [train_[i][1] for i in range(len(train_))]
    train_2 = [train_[i][2] for i in range(len(train_))]
    train_3 = [train_[i][3] for i in range(len(train_))]
    train_4 = [train_[i][4] for i in range(len(train_))]
    train_5 = [train_[i][5] for i in range(len(train_))]
    train_6 = [train_[i][6] for i in range(len(train_))]
    train_data_ = []
    train_data_.append(train_1)
    train_data_.append(train_2)
    train_data_.append(train_3)
    train_data_.append(train_4)
    train_data_.append(train_5)
    train_data_.append(train_6)
    del train_1, train_2, train_3, train_4, train_5, train_6, train

    val = all_data.loc[val_index, :].reset_index()
    val_ = val.to_numpy().tolist()
    val_1 = [val_[i][1] for i in range(len(val_))]
    val_2 = [val_[i][2] for i in range(len(val_))]
    val_3 = [val_[i][3] for i in range(len(val_))]
    val_4 = [val_[i][4] for i in range(len(val_))]
    val_5 = [val_[i][5] for i in range(len(val_))]
    val_6 = [val_[i][6] for i in range(len(val_))]
    val_data_ = []
    val_data_.append(val_1)
    val_data_.append(val_2)
    val_data_.append(val_3)
    val_data_.append(val_4)
    val_data_.append(val_5)
    val_data_.append(val_6)
    del val_1, val_2, val_3, val_4, val_5, val_6, val

    train_dataset = MyDataset(*train_data_)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=1,
                                  pin_memory=True,
                                  collate_fn=collate_fn, drop_last=True)
    val_dataset = MyDataset(*val_data_)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size, num_workers=1,
                                pin_memory=True,
                                collate_fn=collate_fn, drop_last=True)
    for epoch in range(epochs):
      ...
