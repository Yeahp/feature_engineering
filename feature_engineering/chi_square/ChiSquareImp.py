import sys
from feature_engineering.chi_square.ChiCriticalValue import ChiCriticalValue


class ChiSquareImp:

    @staticmethod
    def feature_label_count(filtered_f_values: list):
        '''
        :param filtered_f_values: a list of elements with format as (feature_value, label)
        :return: a list of elements with format as [feature_value, label, count]
        '''
        fea_cnt = []
        filtered_f_values.sort(key=lambda x: (x[0], x[1]), reverse=False)
        i = 0
        item: list = filtered_f_values[0]
        cnt = 0
        while i < len(filtered_f_values):
            if filtered_f_values[i] == item:
                cnt += 1
            else:
                item.append(cnt)
                fea_cnt.append(item)
                cnt = 1
                item = filtered_f_values[i]
            i += 1
        return fea_cnt

    @staticmethod
    def build_cross_table(fea_cnt: list):
        '''
        :param fea_cnt: a list of elements with format as [feature_value, label, count]
        :return: a list of elements with format as (feature_value, [label_1_cnt, label_2_cnt])
        '''
        fea_label_dic = {}
        for record in fea_cnt:
            if record[0] not in fea_label_dic.keys():
                fea_label_dic[record[0]] = [0, 0]
            if record[1] == '0':
                fea_label_dic[record[0]][0] += record[2]
            elif record[1] == '1':
                fea_label_dic[record[0]][1] += record[2]
            else:
                print("data exception: label type error!")
                sys.exit(101)
        return fea_label_dic.items()

    @staticmethod
    def combine_two_feature(fea_label_tuple_a: tuple, fea_label_tuple_b: tuple):
        '''
        :param fea_label_tuple_a: a tuple with value (feature_value, [label_1_cnt, label_2_cnt])
        :param fea_label_tuple_b: a tuple with value (feature_value, [label_1_cnt, label_2_cnt])
        :return: a tuple with value (feature_value, [label_1_cnt, label_2_cnt])
        '''
        fea_label_tuple_c = fea_label_tuple_a[:]
        for i in range(len(fea_label_tuple_a[1])):
            fea_label_tuple_c[1][i] += fea_label_tuple_b[1][i]
        return fea_label_tuple_c

    @staticmethod
    def chi_square_compute(label_count_list):
        '''
        this function is designed to compute chi-square value
        argument A stored as a list of list represents a cross-table
         ---------------------------------
        |         |  label_1  |  label_2  |
         ---------------------------------
        |  f_x_1  |   n_1_1   |   n_1_2   |
         ---------------------------------
        |  f_x_2  |   n_2_1   |   n_2_2   |
         ---------------------------------
        f_x_1 and f_x_2 are feature values while label_1 and label_2 labels,
        and n_1_1, n_1_2, n_2_1 and n_2_2 the number of data points with specific feature value and label
        :param label_count_list: two groups to be compared
        :return: the chi-square value
        '''
        num_record = len(label_count_list)
        num_class = len(label_count_list[0])
        row_sum = []
        for i in range(num_record):
            _row_sum = 0
            for j in range(num_class):
                _row_sum += label_count_list[i][j]
            row_sum.append(_row_sum)
        col_sum = []
        for j in range(num_class):
            _col_sum = 0
            for i in range(num_record):
                _col_sum += label_count_list[i][j]
            col_sum.append(_col_sum)
        total_sum = 0
        for sub_col_sum in col_sum:
            total_sum += sub_col_sum
        chi_square_value = 0
        for i in range(num_record):
            for j in range(num_class):
                exp_i_j = row_sum[i] * col_sum[j] * 1.0 / total_sum
                if exp_i_j > 0:
                    chi_square_value += (label_count_list[i][j] - exp_i_j) ** 2 / exp_i_j
        return chi_square_value

    @staticmethod
    def chi_square_merge(fea_label_tuple: list, alpha: float):
        '''
        :param fea_label_tuple: a list of elements with format as (feature_value, label_1_cnt, label_2_cnt)
        :param alpha: the significant level
        :return: a list of split points
        '''
        num_interval = len(fea_label_tuple)
        freedom = (2 - 1) * (len(fea_label_tuple[0][1]) - 1)
        chi_square_value = ChiCriticalValue.get_critical_value(alpha, freedom)
        while True:
            num_pair = num_interval - 1
            chi_square_values = []
            for i in range(num_pair):
                arr = [fea_label_tuple[i][1], fea_label_tuple[i + 1][1]]
                chi_square_values.append(ChiSquareImp.chi_square_compute(arr))
            for i in range(num_pair - 1, -1, -1):
                if chi_square_values[i] < chi_square_value:
                    fea_label_tuple[i] = ChiSquareImp.combine_two_feature(fea_label_tuple[i], fea_label_tuple[i + 1])
                    fea_label_tuple[i + 1] = 'merged'
            while 'merged' in fea_label_tuple:
                fea_label_tuple.remove('merged')
            if num_interval == len(fea_label_tuple):
                break
            else:
                num_interval = len(fea_label_tuple)
        split_point_list = [record[0] for record in fea_label_tuple]
        return split_point_list
