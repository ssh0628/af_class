import os
import re
import random
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

random.seed(42)

mat_root = r"C:\Users\cream\OneDrive\Desktop\AF_Emu\Data\MAT"

def get_label(filename):
    return 0 if 'non' in filename.lower() else 1

# 사람 ID만 추출
def extract_detailed_person_id(filename):
    # 확장자 .mat 앞에 _data 가 포함된, _data 뒤로는 뭐가 와도 상관없음
    match = re.match(r'(.+?)_(\d{3})_data.*\.mat', filename)
    if match:
        prefix = match.group(1)
        person_num = match.group(2)
        return f"{prefix}_{person_num}"
    else:
        print(f"패턴 불일치 파일명: {filename}")  # 디버그용 출력
        return None


# 1. 사람별 파일과 라벨 수집
person_files = defaultdict(list)
person_labels = defaultdict(list)

for root, _, files in os.walk(mat_root):
    for file in files:
        if file.endswith('.mat'):
            person_id = extract_detailed_person_id(file)
            if person_id is None:
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, mat_root).replace('\\', '/')
            person_files[person_id].append(rel_path)
            # 라벨은 사람 단위로 하나만 저장 (여러 개면 가장 많은 라벨 기준)
            if person_id not in person_labels:
                person_labels[person_id] = []
            person_labels[person_id].append(get_label(file))

# 다수결로 라벨 결정
final_person_labels = {}
for pid, labels in person_labels.items():
    final_person_labels[pid] = round(sum(labels) / len(labels))

# 2. 라벨별 사람 리스트 분리
label_0_persons = [pid for pid, lbl in final_person_labels.items() if lbl == 0]
label_1_persons = [pid for pid, lbl in final_person_labels.items() if lbl == 1]
print(f"Label 0 persons count: {len(label_0_persons)}")
print(f"Label 1 persons count: {len(label_1_persons)}")


# 3. 분할 함수
def split_persons(person_list, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    train, temp = train_test_split(person_list, test_size=(1 - train_ratio), random_state=42)
    valid, test = train_test_split(temp, test_size=test_ratio / (test_ratio + valid_ratio), random_state=42)
    return train, valid, test

train_0, valid_0, test_0 = split_persons(label_0_persons)
train_1, valid_1, test_1 = split_persons(label_1_persons)

train_persons = train_0 + train_1
valid_persons = valid_0 + valid_1
test_persons = test_0 + test_1

# 4. 파일 수집
def collect_files(person_ids):
    files = []
    for pid in person_ids:
        files.extend(person_files[pid])
    return files

train_files = collect_files(train_persons)
valid_files = collect_files(valid_persons)
test_files = collect_files(test_persons)

# 5. 저장
def save_list(filename, files):
    with open(filename, 'w', encoding='utf-8') as f:
        for line in files:
            f.write(f"'{line}'\n")

def print_label_distribution(person_ids, final_person_labels):
    labels = [final_person_labels[pid] for pid in person_ids]
    dist = Counter(labels)
    print(dist)

save_list('train_list.txt', train_files)
save_list('valid_list.txt', valid_files)
save_list('test_list.txt', test_files)

# 6. 정보 출력
print(f"Train persons: {len(train_persons)}, Valid persons: {len(valid_persons)}, Test persons: {len(test_persons)}")
print(f"Train files: {len(train_files)}, Valid files: {len(valid_files)}, Test files: {len(test_files)}")

# 7. 교차 체크
train_set = set(train_persons)
valid_set = set(valid_persons)
test_set = set(test_persons)

print("Train ∩ Valid:", train_set & valid_set)
print("Train ∩ Test :", train_set & test_set)
print("Valid ∩ Test :", valid_set & test_set)

print("Train label distribution:")
print_label_distribution(train_persons, final_person_labels)

print("Valid label distribution:")
print_label_distribution(valid_persons, final_person_labels)

print("Test label distribution:")
print_label_distribution(test_persons, final_person_labels)