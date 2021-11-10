import pandas as pd
import os


def make_one_file_remain_category(merge_dir, individual_dir):
    resultfile = pd.DataFrame()
    merge_files = os.listdir(merge_dir)
    for file in merge_files:
        df_merge = pd.read_excel(merge_dir + "/" + file)
        df_individual_sheet2 = pd.read_excel(individual_dir + "/" + file[8:], sheet_name=1)
        row_size = df_merge['csd'].size

        idx_zero_temp = [i for i in range(1, row_size)]
        df_merge = df_merge.drop(idx_zero_temp)     # 1개 행만 제외하고 제거
        df_merge = df_merge.drop(['time'], axis=1)  # 절대 시간열 제거

        # column - awake & asleep
        df_merge['state'] = 'awake' if df_merge['level'][0] == 'wake' or df_merge['level'][0] == 'awake' or df_merge['level'][0] == 'restless' else 'asleep'
        df_merge = df_merge.drop(['level'], axis=1)  # stage 제거

        # column - age
        df_merge['age'] = df_individual_sheet2['age'][0]

        # column - sex (Binary)
        df_merge['gender'] = df_individual_sheet2['sex'][0]

        # column - height
        df_merge['height'] = df_individual_sheet2['height'][0]

        # column - weight
        df_merge['weight'] = df_individual_sheet2['weight'][0]

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
        df_merge['disease'] = value

        # column - depressive
        df_merge['depressive'] = df_individual_sheet2['q9'][0]

        # column - disorder (Binary)        // 1 : yes,  2: no
        value = "yes" if df_individual_sheet2['q10'][0] == 1 else "no"
        df_merge['disorder'] = value

        # column - media
        value = df_individual_sheet2['q12'][0] - 1
        val_list = [15, 45, 90, 150, 210, 15, 45]
        df_merge['media'] = val_list[value]

        # column - liquor
        value = df_individual_sheet2['dq1'][0] - 1
        val_list = [0, 14, 42, 85, 114]
        df_merge['liquor'] = val_list[value]

        # column - smoke
        value = df_individual_sheet2['dq2'][0] - 1
        val_list = [0, 7, 22, 37, 47]
        df_merge['smoke'] = val_list[value]

        # column - caffeine
        value = df_individual_sheet2['dq3'][0] - 1
        val_list = [0, 189, 30, 43, 65]
        df_merge['caffeine'] = val_list[value]

        # column - exercise
        value = df_individual_sheet2['dq4'][0] - 1
        val_list = [0, 15, 45, 90, 150]
        df_merge['exercise'] = val_list[value]

        # column - stress
        df_merge['stress'] = df_individual_sheet2['dq5'][0]

        # column - nap (Binary)
        val = "yes" if df_individual_sheet2['dq6'][0] == 1 else "no"
        df_merge['nap'] = val

        resultfile = pd.concat([resultfile, df_merge])

    print(resultfile)
    return resultfile


result = make_one_file_remain_category("valid_merge", "valid_individual")
result.to_csv('real_validation.csv', index=False)


##df_sheet = pd.read_excel("sleep_merge/new_2107200321_박동한_1626744303108.xls", sheet_name=0)
##df_sheet1 = pd.read_excel("sleep_merge/new_2107210354_박동한_1626826441567.xls", sheet_name=0)
##print(df_sheet)
##print(df_sheet1)

##df_row = pd.concat([df_sheet, df_sheet1])
##print(df_row)
