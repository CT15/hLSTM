import pandas as pd

data = pd.read_csv('./data/sorted_result_large.csv')

unique_ids = set(data.thread_id)
size = len(unique_ids)

# ones = len(data[data.labels == 1])
# zeros = len(data[data.labels == 0])
# print(len(data), ones, zeros, ones/zeros)

count_1 = int(round(1.0 / 100 * size))
count_5 = int(round(5.0 / 100 * size))
count_10 = int(round(10.0 / 100 * size))
count_30 = int(round(30.0 / 100 * size))
count_50 = int(round(50.0 / 100 * size))
count_75 = int(round(75.0 / 100 * size))

count = -1
go_in = False
prev_id = 'dummy_id'
for index, row in data.iterrows():
    
    thread_id = row['thread_id']

    if prev_id != thread_id:
        count += 1
        prev_id = thread_id
        go_in = True

    cmin1 = count - 1
    if go_in and (cmin1 == count_1 or cmin1 == count_5 or 
        cmin1 == count_10 or cmin1 == count_30 or 
        cmin1 == count_50 or cmin1 == count_75):

        go_in = False
        df = data.iloc[:index+1]
        
        if cmin1 == count_1:
            print('1%')
            df.to_csv('./data/result_large_1.csv')
        if cmin1 == count_5:
            print('5%')
            df.to_csv('./data/result_large_5.csv', index=False)
        if cmin1 == count_10:
            print('10%')
            df.to_csv('./data/result_large_10.csv', index=False)
        if cmin1 == count_30:
            print('30%')
            df.to_csv('./data/result_large_30.csv', index=False)
        if cmin1 == count_50:
            print('50%')
            df.to_csv('./data/result_large_50.csv', index=False)
        if cmin1 == count_75:
            print('75%')
            df.to_csv('./data/result_large_75.csv', index=False)

        print(f'no of threads = {cmin1}')
        print(f'data points = {index+1}')
        ones = len(df[df.labels == 1])
        zeros = len(df[df.labels == 0])
        print(f'1s = {ones}')
        print(f'0s = {zeros}')
        print(f'ratio = {ones/(zeros+ones)}')
        print('----------')
