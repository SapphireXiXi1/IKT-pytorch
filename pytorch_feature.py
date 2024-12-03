import torch
import pandas as pd
import csv
from BKT import BKT
import numpy as np

data_name = '4_Ass_09'
# data_name = '4_algebra05'
# data_name = '4_Ass_12'
batch_size = 256


def read_data_from_csv_file(trainfile, testfile, device):
    rows = []  # 用于存储从 CSV 文件读取的原始数据
    max_skills = 0  # 记录数据中出现的最大知识点编号
    max_steps = 0  # 记录学生解答的最大题目数量（分段后可能扩展）
    max_items = 0  # 记录数据中题目的最大编号
    studentids = []  # 存储所有学生的 ID
    train_ids = []  # 存储训练集中的学生 ID
    test_ids = []  # 存储测试集中的学生 ID
    problem_len = 20  # 指定分段长度为 20

    # 打开训练集文件，使用逗号作为分隔符，按行读取数据，每行数据存储为 row，将其内容按行存储到 rows 中。
    with open(trainfile, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)

    skill_rows = torch.tensor([], dtype=torch.long, device=device)  # 知识点编号的序列
    correct_rows = torch.tensor([], dtype=torch.float32, device=device)  # 答题正确性（0 或 1）
    stu_rows = torch.tensor([], dtype=torch.long, device=device)  # 学生 ID
    opp_rows = torch.tensor([], dtype=torch.long, device=device)  # 题目序号（每位学生答题序列中的位置）

    # 处理训练集数据
    index = 0
    while index < len(rows):
        if int(rows[index][0]) > problem_len:
            problems = int(rows[index][0])  # 当前学生的答题总数
            student_id = int(rows[index][1])  # 当前学生的 ID
            train_ids.append(student_id)  # 将学生 ID 加入训练集学生 ID 列表

            tmp_max_skills = max(map(int, rows[index + 1]))  # 当前学生答题涉及的最大知识点编号
            max_skills = max(max_skills, tmp_max_skills)  # 更新全局最大知识点编号

            tmp_max_items = max(map(int, rows[index + 2]))  # 当前学生答题涉及的最大题目编号
            max_items = max(max_items, tmp_max_items)  # 更新全局最大题目编号

            skill_rows = torch.cat((skill_rows, torch.tensor(list(map(int, rows[index + 1])), dtype=torch.long, device=device)))
            correct_rows = torch.cat((correct_rows, torch.tensor(list(map(int, rows[index + 3])), dtype=torch.float32, device=device)))
            stu_rows = torch.cat((stu_rows, torch.tensor([student_id] * len(rows[index + 1]), dtype=torch.long, device=device)))
            opp_rows = torch.cat((opp_rows, torch.arange(len(rows[index + 1]), dtype=torch.long, device=device)))
        index += 4

    # 读取测试集文件
    with open(testfile, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)

    while index < len(rows):
        if int(rows[index][0]) > problem_len:
            problems = int(rows[index][0])
            student_id = int(rows[index][1])
            test_ids.append(student_id)

            tmp_max_skills = max(map(int, rows[index + 1]))
            max_skills = max(max_skills, tmp_max_skills)

            tmp_max_items = max(map(int, rows[index + 2]))
            max_items = max(max_items, tmp_max_items)

            skill_rows = torch.cat((skill_rows, torch.tensor(list(map(int, rows[index + 1])), dtype=torch.long, device=device)))
            correct_rows = torch.cat((correct_rows, torch.tensor(list(map(int, rows[index + 3])), dtype=torch.float32, device=device)))
            stu_rows = torch.cat((stu_rows, torch.tensor([student_id] * len(rows[index + 1]), dtype=torch.long, device=device)))
            opp_rows = torch.cat((opp_rows, torch.arange(len(rows[index + 1]), dtype=torch.long, device=device)))
        index += 4

    # 因为知识点编号和题目编号是从 0 开始的，最大值需要加 1，确保范围完整
    max_skills += 1
    max_items += 1

    # 构建数据框
    data = pd.DataFrame({
        'stus': stu_rows.tolist(),
        'skills': skill_rows.tolist(),
        'corrects': correct_rows.tolist(),
        'opp': opp_rows.tolist()
    }).astype(int)

    # 假设 BKTAssessment 替换为 PyTorch 实现或 PyTorch 兼容操作
    bkt_ass = BKTAssessment(data, train_ids, max_skills, device=device)

    del skill_rows, correct_rows, stu_rows, opp_rows, data  # 删除前面用过的数据存储列表，释放内存。

    # 处理完整学生数据并进行分段
    index = 0
    tuple_rows = []
    while index < len(rows):
        if int(rows[index][0]) > problem_len:
            problems = int(rows[index][0])  # 当前学生的答题总数
            student_id = int(rows[index][1])  # 当前学生的 ID
            studentids.append(student_id)  # 将学生 ID 加入所有学生 ID 列表

            if problems > problem_len:
                tmp_max_steps = problems  # 这个学生解答的最大题目数量
                max_steps = max(max_steps, tmp_max_steps)  # 更新全局最大题目数量

                asses = bkt_ass[student_id]  # 根据学生 ID 获取学生知识掌握的评估结果

                len_problems = (problems // problem_len) * problem_len
                rest_problems = problems - len_problems

                ele_p = torch.tensor(list(map(int, rows[index + 1])), dtype=torch.long, device=device)
                ele_c = torch.tensor(list(map(int, rows[index + 3])), dtype=torch.float32, device=device)
                ele_d = torch.tensor(list(map(int, rows[index + 2])), dtype=torch.long, device=device)
                ele_a = torch.tensor(asses, dtype=torch.float32, device=device)

                if rest_problems > 0:
                    rest = problem_len - rest_problems
                    ele_p = torch.cat((ele_p, torch.full((rest,), -1, dtype=torch.long, device=device)))
                    ele_c = torch.cat((ele_c, torch.full((rest,), -1, dtype=torch.float32, device=device)))
                    ele_d = torch.cat((ele_d, torch.full((rest,), -1, dtype=torch.long, device=device)))
                    ele_a = torch.cat((ele_a, torch.full((rest,), -1, dtype=torch.float32, device=device)))

                ele_p_array = ele_p.view(-1, problem_len)
                ele_c_array = ele_c.view(-1, problem_len)
                ele_d_array = ele_d.view(-1, problem_len)
                ele_a_array = ele_a.view(-1, problem_len)

                n_pieces = ele_p_array.size(0)

                for j in range(n_pieces):
                    s1 = [student_id, j, problems]
                    if j < n_pieces - 1:
                        s1.append(1)
                        s2 = torch.cat((ele_p_array[j], ele_p_array[j + 1, :1])).tolist()
                        s3 = torch.cat((ele_c_array[j], ele_c_array[j + 1, :1])).tolist()
                        s4 = torch.cat((ele_d_array[j], ele_d_array[j + 1, :1])).tolist()
                        s5 = torch.cat((ele_a_array[j], ele_a_array[j + 1, :1])).tolist()
                    else:
                        s1.append(-1)
                        s2 = ele_p_array[j].tolist()
                        s3 = ele_c_array[j].tolist()
                        s4 = ele_d_array[j].tolist()
                        s5 = ele_a_array[j].tolist()
                    tup = (s1, s2, s3, s4, s5)
                    tuple_rows.append(tup)
        index += 4

    max_steps += 1

    # 按学生划分训练集与测试集
    train_students = [tr for tr in tuple_rows if tr[0][0] in train_ids]
    test_students = [ts for ts in tuple_rows if ts[0][0] in test_ids]

    return train_students, test_students, studentids, max_skills, max_items, train_ids, test_ids


def BKTAssessment(data, train_ids, max_skills, device):
    # bkt_data：一个字典，每个技能对应一个子字典，其中每位学生的答题序列被记录
    # dkt_skill 和 dkt_res：用于预测的技能序列和学生答题序列
    bkt_data, dkt_skill, dkt_res = get_bktdata(data, device)

    # 存储 BKT 模型的四个核心参数：DL：初始掌握概率（P(K0)）；DT：转移概率（P(T)）；DG：猜测概率（P(G)）；DS：错误率（P(S)）
    DL, DT, DG, DS = {}, {}, {}, {}

    for skill_id in bkt_data.keys():
        skill_data = bkt_data[skill_id]  # 当前技能的答题记录
        train_data = []

        for student_id in skill_data.keys():
            if int(student_id) in train_ids:
                train_data.append(skill_data[student_id].to(device))  # 将训练数据移到 GPU

        # 初始化 BKT 模型
        bkt = BKT(step=0.1, bounded=False, best_k0=True)


        if len(train_data) > 2:  # 如果训练数据足够
            DL[skill_id], DT[skill_id], DG[skill_id], DS[skill_id] = bkt.fit(train_data)
        else:  # 否则使用默认值
            DL[skill_id], DT[skill_id], DG[skill_id], DS[skill_id] = 0.5, 0.2, 0.1, 0.1

    del bkt_data  # 清理无用数据

    # 使用模型参数计算每位学生对每个技能的掌握情况
    mastery = bkt.inter_predict(dkt_skill, dkt_res, DL, DT, DG, DS, max_skills)

    del dkt_skill, dkt_res  # 清理中间变量以节省内存

    print("**************Finished BKT Assessment****************")
    return mastery


def get_bktdata(df, device):
    # 初始化三个字典
    BKT_dict = {}  # 用于存储按知识点划分的学生答题结果
    DKT_skill_dict = {}  # 用于存储按学生划分的答题技能序列
    DKT_res_dict = {}  # 用于存储按学生划分的答题结果序列

    # 构建 BKT 数据结构
    for kc in df['skills'].unique():  # 提取数据中所有的知识点（技能 ID）
        kc_df = df[df['skills'] == kc].sort_values('stus')  # 按学生 ID 排序
        stu_cfa_dict = {}  # 存储当前知识点下每个学生的答题结果序列

        for stu in kc_df['stus'].unique():  # 遍历当前知识点下的所有学生 ID
            df_final = kc_df[kc_df['stus'] == int(stu)].sort_values('opp')  # 按答题机会排序
            stu_cfa_dict[int(stu)] = torch.tensor(df_final['corrects'].tolist(), dtype=torch.float32, device=device)  # 转为 PyTorch Tensor，并移到 GPU

        BKT_dict[int(kc)] = stu_cfa_dict  # 存储当前知识点的学生答题记录

    # 构建 DKT 数据结构
    for stu in df['stus'].unique():  # 遍历所有学生 ID
        stu_df = df[df['stus'] == int(stu)].sort_values('opp')  # 按答题机会排序
        DKT_skill_dict[int(stu)] = torch.tensor(stu_df['skills'].tolist(), dtype=torch.int64, device=device)  # 转为 PyTorch Tensor，并移到 GPU
        DKT_res_dict[int(stu)] = torch.tensor(stu_df['corrects'].tolist(), dtype=torch.float32, device=device)  # 转为 PyTorch Tensor，并移到 GPU

    return BKT_dict, DKT_skill_dict, DKT_res_dict


# 难度计算
def difficulty_data(students, max_items, device):
    limit = 3  # 计算题目难度时所需的最小回答次数
    xtotal = torch.zeros(max_items + 1, dtype=torch.float32, device=device)  # 每个题目被回答的总次数
    x1 = torch.zeros(max_items + 1, dtype=torch.float32, device=device)  # 每个题目回答错误的次数
    items = []  # 符合条件的题目 ID
    Allitems = []  # 所有题目 ID
    item_diff = {}  # 每个题目的难度分数字典

    # 遍历学生答题记录
    for student in students:
        item_ids = student[3]  # 学生答题的题目 ID
        correctness = student[2]  # 学生答题的正确性

        # 遍历学生答题的每个题目
        for j in range(len(item_ids)):
            key = item_ids[j]  # 当前题目 ID
            xtotal[key] += 1  # 累加题目被回答的总次数
            if int(correctness[j]) == 0:  # 如果回答错误
                x1[key] += 1

            # 检查题目是否符合条件
            if xtotal[key] > limit and key > 0 and key not in items:
                items.append(key)
            if xtotal[key] > 0 and key not in Allitems:
                Allitems.append(key)

    # 计算符合条件题目的难度
    for i in items:
        # 计算错误率并转换为整数分数
        diff = (torch.round(x1[i] / xtotal[i] * 10)).int().item()
        item_diff[i] = diff  # 存储题目 ID 和对应的难度分数

    return item_diff


# 聚类计算
def cluster_data(students, max_stu, num_skills, datatype, batch_size, device):
    success = []  # 存储每个学生的特征数据（成功率、学生 ID、分段编号）
    max_seg = 0  # 所有学生中分段编号的最大值
    xtotal = torch.zeros((max_stu, num_skills), dtype=torch.float32, device=device)  # 学生对每个技能的总尝试次数
    x1 = torch.zeros((max_stu, num_skills), dtype=torch.float32, device=device)  # 学生对每个技能的答对次数
    x0 = torch.zeros((max_stu, num_skills), dtype=torch.float32, device=device)  # 学生对每个技能的答错次数

    # 遍历学生数据（按批次处理）
    index = 0
    while index + batch_size < len(students):
        for i in range(batch_size):  # 处理当前批次内的学生数据
            student = students[index + i]
            student_id = int(student[0][0])
            seg_id = int(student[0][1])

            # 更新分段最大值
            if int(student[0][3]) == 1:  # 检查条件
                tmp_seg = seg_id
                if tmp_seg > max_seg:
                    max_seg = tmp_seg

                # 统计技能掌握数据
                problem_ids = student[1]  # 学生回答的问题 ID
                correctness = student[2]  # 问题回答的正确性
                for j in range(len(problem_ids)):
                    key = problem_ids[j]
                    xtotal[student_id, key] += 1  # 更新总尝试次数
                    if int(correctness[j]) == 1:
                        x1[student_id, key] += 1  # 答对次数
                    else:
                        x0[student_id, key] += 1  # 答错次数

                # 计算成功率
                xsr = ((x1[student_id] + 1.4) / (xtotal[student_id] + 2)).tolist()  # 使用平滑项计算成功率

                # 组合特征数据
                x = torch.tensor(xsr, dtype=torch.float32, device=device)  # 成功率向量
                x = torch.cat([x, torch.tensor([student_id], dtype=torch.float32, device=device), torch.tensor([seg_id], dtype=torch.float32, device=device)])  # 添加学生 ID 和分段编号
                success.append(x.cpu().numpy())  # 将结果转为 NumPy 数组并添加到 success 列表中

        index += batch_size  # 更新批次索引

    return success, max_seg  # 返回特征数据和最大分段编号


def k_means_clust(train_students, test_students, max_stu, max_seg, num_clust, num_skills, num_iter, device):
    identifiers = 3  # 标志用于去除非特征列，如学生 ID、分段编号等。
    max_stu = int(max_stu)  # 学生 ID 的最大值
    max_seg = int(max_seg)  # 分段编号的最大值
    cluster = torch.zeros((max_stu, max_seg), dtype=torch.long, device=device)  # 用于存储最终的聚类结果
    data = []  # 存储训练数据的特征向量

    # 构建聚类输入数据
    for i in train_students:
        data.append(i[:-identifiers])  # 提取特征部分（去掉 identifiers 列，如学生 ID 和分段编号）
    data = torch.tensor(data, dtype=torch.float32, device=device)  # 转换为 PyTorch tensor并移动到GPU

    # 初始化聚类中心
    centroids = data[torch.randint(0, data.size(0), (num_clust,), device=device)]  # 随机选择 num_clust 个点作为初始聚类中心

    # 迭代训练
    for _ in range(num_iter):
        # 计算每个点到所有聚类中心的距离
        distances = torch.cdist(data, centroids)  # 计算欧几里得距离 (N, k)
        indices = torch.argmin(distances, dim=1)  # (N,) 每个点属于最近的聚类中心

        # 更新聚类中心
        new_centroids = torch.stack([data[indices == i].mean(dim=0) for i in range(num_clust)])  # 计算每个聚类的均值作为新的聚类中心

        # 检查聚类中心是否收敛
        if torch.allclose(new_centroids, centroids, atol=1e-4):
            break
        centroids = new_centroids

    # 对训练集分段编号的聚类结果映射
    for i in train_students:
        inst = torch.tensor(i[:-identifiers], dtype=torch.float32, device=device)  # 提取特征部分并移动到GPU
        min_dist = float('inf')
        closest_clust = None
        for j in range(num_clust):
            cur_dist = euclidean_distance(inst, centroids[j])  # 计算当前点到第 j 个聚类中心的欧几里得距离
            if cur_dist < min_dist:
                min_dist = cur_dist
                closest_clust = j
        cluster[int(i[-2]), int(i[-1])] = closest_clust  # 保存聚类结果

    # 对测试集分段编号的聚类结果映射
    for i in test_students:
        inst = torch.tensor(i[:-identifiers], dtype=torch.float32, device=device)  # 提取特征部分并移动到GPU
        min_dist = float('inf')
        closest_clust = None
        for j in range(num_clust):
            cur_dist = euclidean_distance(inst, centroids[j])  # 计算当前点到第 j 个聚类中心的欧几里得距离
            if cur_dist < min_dist:
                min_dist = cur_dist
                closest_clust = j
        cluster[int(i[-2]), int(i[-1])] = closest_clust  # 保存聚类结果

    return cluster  # 返回学生在各分段的聚类结果


# 欧几里得距离计算（向量化方式），未修改
def euclidean_distance(instance1, instance2):
    instance1 = torch.tensor(instance1, dtype=torch.float32)
    instance2 = torch.tensor(instance2, dtype=torch.float32)
    distance = torch.sqrt(torch.sum((instance1 - instance2) ** 2))
    return distance


# 获取特征并生成输入TAN的特征数据
def get_features(students, item_diff, max_stu, cluster, num_skills, datatype, batch_size, device):
    """运行模型并生成特征数据"""
    index = 0  # 当前处理到的学生索引

    # 初始化列表用于存储学生特征
    stu_list = []  # 学生 ID
    p0_list = []  # 知识点（技能）ID
    p1_list = []  # 学生对知识点的掌握程度
    p2_list = []  # 学生的分段簇信息
    p3_list = []  # 题目难度
    p4_list = []  # 答题正确与否
    total_students = len(students)  # 学生总数

    # 逐批处理学生数据，确保不会越界
    while (index + batch_size < len(students)):

        print(f"共有学生 {total_students}, 正在处理第 {index} 至 {index + batch_size} 个学生...")

        for i in range(batch_size):  # 遍历当前批次的学生
            student = students[index + i]  # 获取当前学生数据
            student_id = student[0][0]  # 学生 ID
            seg_id = int(student[0][1])  # 学生当前分段编号，用于查找分段簇信息

            ## 获取学生在前一段的簇类别
            if (seg_id > 0):
                cluster_id = cluster[student_id, (seg_id - 1)] + 1  # 获取上一段的簇类别
            else:
                cluster_id = 0  # 如果是第一段，簇类别默认为 0

            skill_ids = student[1]  # 知识点序列
            correctness = student[2]  # 答题正确性序列
            items = student[3]  # 题目 ID 序列
            bkt = student[4]  # 知识点掌握程度（来自 BKT 模型）

            for j in range(len(skill_ids) - 1):  # 遍历学生的答题序列（跳过第一个数据）
                target_indx = j + 1  # 目标答题的索引
                skill_id = int(skill_ids[target_indx])  # 知识点 ID
                item = int(items[target_indx])  # 题目 ID
                kcass = np.round(float(bkt[target_indx]), 6)  # 知识点掌握程度（取小数点后6位）

                correct = int(correctness[target_indx])  # 答题结果（0 或 1）

                if skill_id > -1:  # 仅处理有效的知识点（排除 -1 等无效标记）
                    df = 0
                    if item in item_diff.keys():  # 如果题目在 item_diff 字典中
                        df = int(item_diff[item])  # 使用题目的难度值
                    else:
                        df = 5  # 默认难度值为 5

                    # 将提取的特征保存到对应的列表中
                    stu_list.append(student_id)  # 学生 ID
                    p0_list.append(int(skill_id))  # 知识点 ID
                    p1_list.append(float(kcass))  # 知识点掌握程度
                    p2_list.append(int(cluster_id))  # 学生的分段簇信息
                    p3_list.append(int(df))  # 题目难度
                    p4_list.append(int(correct))  # 答题正确性

        index += batch_size  # 移动到下一个批次

    # 将所有提取的特征数据构建成一个 PyTorch 张量，并移至 GPU
    data_tensor = torch.tensor([stu_list, p0_list, p1_list, p2_list, p3_list, p4_list], dtype=torch.int64, device=device).T

    # 将 PyTorch 张量转回 CPU，并转换为 pandas 数据框以便保存
    data_df = pd.DataFrame(data_tensor.cpu().numpy(), columns=['student_id', 'skill_id', 'skill_mastery', 'ability_profile', 'problem_difficulty', 'correctness'])

    # 保存为 CSV 文件
    data_df.to_csv(f"./data_convert_result/{datatype}_data.csv", index=False, header=True)

    return


# 主要逻辑函数
def main():
    # 设置设备为 GPU，如果没有 GPU 则使用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'GPU是否可用：{torch.cuda.is_available()}')

    # 设置聚类数量、每个学生的时间序列长度（题目数量）
    cluster_num = 7
    problem_len = 20

    # 读取训练数据文件和测试数据文件路径
    train_data = './data/' + data_name + '_train.csv'
    test_data = './data/' + data_name + '_test.csv'

    # 读取训练和测试数据文件，提取相关信息
    # 在这里添加device参数
    print(f'read_data_from_csv_file')
    train_students, test_students, student_ids, max_skills, max_items, train_ids, test_ids = read_data_from_csv_file(
        train_data, test_data, device)  # 传递device

    num_skills = max_skills

    # 计算问题难度
    print(f'计算问题难度')
    item_diff = difficulty_data(train_students + test_students, max_items)

    # 对学生行为数据进行聚类：根据学生的答题行为生成用于聚类的数据
    print(f'cluster_data')
    train_cluster_data, train_max_seg = cluster_data(train_students, max(train_ids) + 1, max_skills, "train")
    test_cluster_data, test_max_seg = cluster_data(test_students, max(test_ids) + 1, max_skills, "test")

    # max_stu：学生的总数（取最大学生ID加1）
    # max_seg：分段的总数量，取训练和测试的最大值 +1，确保统一长度
    max_stu = max(student_ids) + 1
    max_seg = max([int(train_max_seg), int(test_max_seg)]) + 1

    # 使用GPU的K-means聚类
    print(f'k_means_clust')
    cluster = k_means_clust(train_cluster_data, test_cluster_data, max_stu, max_seg, cluster_num,
                            max_skills, 40, device=device)  # 确保cluster函数也使用device

    # 提取训练集和测试集的特征
    print(f'get_features')
    get_features(train_students, item_diff, max_stu, cluster, max_skills, "train", batch_size=32, device=device)
    get_features(test_students, item_diff, max_stu, cluster, max_skills, "test", batch_size=32, device=device)


# 如果需要在命令行运行主函数
if __name__ == "__main__":
    main()