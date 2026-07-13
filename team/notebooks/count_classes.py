# -*- coding: utf-8 -*-
import os
from collections import Counter
from pathlib import Path

current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent.parent

# ==================== 경로 설정 ====================
label_dir = str(project_root / "team" / "data" / "raw" / "acai_basic_data" / "yolo_dataset" / "final_split" / "labels" / "train")

# ==================== 알약 이름 매핑 리스트 ====================
# [1] YOLO 학습용 data.yaml에 복사할 체계와 100% 동기화
CLASS_NAMES = {
    0: "마그밀정(수산화마그네슘)",
    1: "게보린정 300mg/PTP",
    2: "알마겔정(알마게이트)(수출명:유한가스트라겔정)",
    3: "보령부스파정 5mg",
    4: "뮤테란캡슐 100mg",
    5: "일양하이트린정 2mg",
    6: "기넥신에프정(은행엽엑스)(수출용)",
    7: "무코스타정(레바미피드)(비매품)",
    8: "동아오팔몬정(리마프로스트알파-시클로덱스트린포접화합물)",
    9: "알드린정",
    10: "뉴로메드정(옥시라세탐)",
    11: "타이레놀정500mg",
    12: "에어탈정(아세클로페낙)",
    13: "비유피-4정 20mg",
    14: "엘도스캡슐(에르도스테인)(수출용)",
    15: "삼남건조수산화알루미늄겔정",
    16: "프로스카정",
    17: "타이레놀이알서방정(아세트아미노펜)(수출용)",
    18: "삐콤씨에프정 618.6mg/병",
    19: "조인스정 200mg",
    20: "쎄로켈정 100mg",
    21: "넥시움정 40mg",
    22: "아스피린프로텍트정 100mg",
    23: "리렉스펜정 300mg/PTP",
    24: "아빌리파이정 10mg",
    25: "자이프렉사정 2.5mg",
    26: "다보타민큐정 10mg/병",
    27: "엘스테인캡슐(에르도스테인)",
    28: "써스펜8시간이알서방정 650mg",
    29: "에빅사정(메만틴염산염)(비매품)",
    30: "한미탐스캡슐 0.2mg",
    31: "아보다트연질캡슐 0.5mg",
    32: "리피토정 20mg",
    33: "크레스토정 20mg",
    34: "가바토파정 100mg",
    35: "동아가바펜틴정 800mg",
    36: "오마코연질캡슐(오메가-3-산에틸에스테르90)",
    37: "란스톤엘에프디티정 30mg",
    38: "리리카캡슐 150mg",
    39: "종근당글리아티린연질캡슐(콜린알포세레이트)",
    40: "콜리네이트연질캡슐 400mg",
    41: "트루비타정 60mg/병",
    42: "스토가정 10mg",
    43: "노바스크정 5mg",
    44: "마도파정",
    45: "플라빅스정 75mg",
    46: "자트랄엑스엘정 10mg",
    47: "베시케어정 10mg",
    48: "엑스포지정 5/160mg",
    49: "펠루비정(펠루비프로펜)",
    50: "아토르바정 10mg",
    51: "라비에트정 20mg",
    52: "리피로우정 20mg",
    53: "자누비아정 50mg",
    54: "맥시부펜이알정 300mg",
    55: "메가파워정 90mg/병",
    56: "쿠에타핀정 25mg",
    57: "비타비백정 100mg/병",
    58: "토비애즈서방정 4mg",
    59: "놀텍정 10mg",
    60: "자누메트정 50/850mg",
    61: "큐시드정 31.5mg/PTP",
    62: "아모잘탄정 5/100mg",
    63: "세비카정 10/40mg",
    64: "트윈스타정 40/5mg",
    65: "카나브정 60mg",
    66: "울트라셋이알서방정",
    67: "졸로푸트정 100mg",
    68: "플리바스정 50mg",
    69: "트라젠타정(리나글립틴)",
    70: "비모보정 500/20mg",
    71: "레일라정",
    72: "리바로정 4mg",
    73: "렉사프로정 15mg",
    74: "트라젠타듀오정 2.5/850mg",
    75: "낙소졸정 500/20mg",
    76: "아질렉트정(라사길린메실산염)",
    77: "자누메트엑스알서방정 100/1000mg",
    78: "글리아타민연질캡슐",
    79: "신바로정",
    80: "트루패스정 4mg",
    81: "에스원엠프정 20mg",
    82: "브린텔릭스정 20mg",
    83: "글리틴정(콜린알포세레이트)",
    84: "제미메트서방정 50/1000mg",
    85: "아토젯정 10/40mg",
    86: "로수젯정10/5밀리그램",
    87: "알바스테인캡슐(에르도스테인)",
    88: "로수바미브정 10/20mg",
    89: "뮤코원캡슐(에르도스테인)",
    90: "카발린캡슐 25mg",
    91: "케이캡정 50mg",
    92: "엘스테인정(에르도스테인)"
}


def count_class_instances():
    if not os.path.exists(label_dir):
        print(f"❌ 경로를 찾을 수 없습니다: {label_dir}")
        return

    class_counter = Counter()
    total_files = 0
    total_instances = 0

    print("🔍 [YOLO 라벨 정밀 분석 프로세스 가동]...\n")

    # 모든 txt 파일 전수조사
    for filename in os.listdir(label_dir):
        if not filename.lower().endswith('.txt'):
            continue

        total_files += 1
        filepath = os.path.join(label_dir, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])  # 문자열을 정수형 번호로 변경
                        class_counter[class_id] += 1
                        total_instances += 1
        except Exception as e:
            print(f"🔺 파일 읽기 오류 ({filename}): {str(e)}")

    # ==================== 결과 출력 ====================
    print("📊 [데이터셋 클래스 분포 결과]")
    print("=" * 65)
    print(f"📂 분석 대상 폴더 : {label_dir}")
    print(f"📸 총 라벨 파일 수 : {total_files}장")
    print(f"💊 총 발견된 알약  : {total_instances}개")
    print("=" * 65)
    print(f"{'클래스 ID':<8} | {'알약 제품명':<25} | {'개수':<6} | {'비율(%)'}")
    print("-" * 65)

    # 0번부터 55번까지 순차적으로 정렬하여 출력 (데이터 상태 직관적 파악 가능)
    for i in range(93):
        count = class_counter[i]
        pill_name = CLASS_NAMES.get(i, "정의되지 않은 클래스")
        percentage = (count / total_instances) * 100 if total_instances > 0 else 0

        # 데이터가 아예 없는 빈 클래스는 눈에 띄게 표기
        if count == 0:
            print(f"ID {i:<5} | {pill_name:<30} | {count:<6} | {percentage:.1f}% (⚠️ 데이터 없음)")
        else:
            print(f"ID {i:<5} | {pill_name:<30} | {count:<6} | {percentage:.1f}%")

    print("=" * 65)

    # 불균형 상태 경고 시스템
    if class_counter:
        most_common_id, most_count = class_counter.most_common(1)[0]
        most_name = CLASS_NAMES.get(most_common_id, "Unknown")
        if (most_count / total_instances) > 0.5:
            print(f"🚨 위험 경고: '{most_name}(ID {most_common_id})' 제품이 전체의 50%를 초과한 심각한 쏠림 상태입니다!")


if __name__ == '__main__':
    count_class_instances()