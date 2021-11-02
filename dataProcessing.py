import pandas as pd
import os


def make_one_file_remain_category(merge_dir, individual_dir):
    resultfile = pd.DataFrame()
    merge_files = os.listdir(merge_dir)
    for file in merge_files:
        df_merge = pd.read_excel(merge_dir + "/" + file)
        df_individual_sheet2 = pd.read_excel(individual_dir + "/" + file[8:], sheet_name=1)
        row_size = df_merge['csd'].size
        df_merge = df_merge.drop(['time'], axis=1)  # 절대 시간열 제거
        # df_merge['time'] = [i for i in range(row_size)]  # 상대 시간열 추가

        # column - awake & asleep
        val_list = list()
        for i in range(row_size):
            if df_merge['level'][i] == 'wake' or df_merge['level'][i] == 'awake' or df_merge['level'][i] == 'restless':
                val_list.append('awake')
            else:
                val_list.append('asleep')
        df_merge['state'] = val_list
        df_merge = df_merge.drop(['level'], axis=1)  # stage 제거

        # column - age
        value = df_individual_sheet2['age'][0]
        df_merge['age'] = [value for i in range(row_size)]

        # column - sex (Binary)
        value = df_individual_sheet2['sex'][0]
        df_merge['gender'] = [value for i in range(row_size)]

        # column - height
        value = df_individual_sheet2['height'][0]
        df_merge['height'] = [value for i in range(row_size)]

        # column - weight
        value = df_individual_sheet2['weight'][0]
        df_merge['weight'] = [value for i in range(row_size)]

        # column - disease (5 Category) -> 처리잘해야할듯 (다중 선택을 최초선택한 하나만으로 설정)
        value = df_individual_sheet2['q5'][0]
        value = str(value)[0]
        if value == '0':
            value = "none"
        elif value == '1':
            value = "hypertensive"
        elif value == '2':
            value = "diabetes"
        elif value == '3':
            value = "liverDisease"
        elif value == '4':
            value = "tuberculosis"
        elif value == '5':
            value = "mental"
        df_merge['disease'] = [value for i in range(row_size)]

        # column - depressive
        value = df_individual_sheet2['q9'][0]
        df_merge['depressive'] = [value for i in range(row_size)]

        # column - disorder (Binary)        // 1 : yes,  2: no
        value = "yes" if df_individual_sheet2['q10'][0] == 1 else "no"
        df_merge['disorder'] = [value for i in range(row_size)]

        # column - media
        value = df_individual_sheet2['q12'][0] - 1
        val_list = [15, 45, 90, 150, 210, 15, 45]
        df_merge['media'] = [val_list[value] for i in range(row_size)]

        # column - liquor
        value = df_individual_sheet2['dq1'][0] - 1
        val_list = [0, 14, 42, 85, 114]
        df_merge['liquor'] = [val_list[value] for i in range(row_size)]

        # column - smoke
        value = df_individual_sheet2['dq2'][0] - 1
        val_list = [0, 7, 22, 37, 47]
        df_merge['smoke'] = [val_list[value] for i in range(row_size)]

        # column - caffeine
        value = df_individual_sheet2['dq3'][0] - 1
        val_list = [0, 189, 30, 43, 65]
        df_merge['caffeine'] = [val_list[value] for i in range(row_size)]

        # column - exercise
        value = df_individual_sheet2['dq4'][0] - 1
        val_list = [0, 15, 45, 90, 150]
        df_merge['exercise'] = [val_list[value] for i in range(row_size)]

        # column - stress
        value = df_individual_sheet2['dq5'][0]
        df_merge['stress'] = [value for i in range(row_size)]

        # column - nap (Binary)
        val = "yes" if df_individual_sheet2['dq6'][0] == 1 else "no"
        df_merge['nap'] = [val for i in range(row_size)]

        resultfile = pd.concat([resultfile, df_merge])

    print(resultfile)
    return resultfile


result = make_one_file_remain_category("sleep_merge", "sleep_individual")
result.to_csv('sample.csv', index=False)

##df_sheet = pd.read_excel("sleep_merge/new_2107200321_박동한_1626744303108.xls", sheet_name=0)
##df_sheet1 = pd.read_excel("sleep_merge/new_2107210354_박동한_1626826441567.xls", sheet_name=0)
##print(df_sheet)
##print(df_sheet1)

##df_row = pd.concat([df_sheet, df_sheet1])
##print(df_row)
