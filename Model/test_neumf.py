# test_neumf.py
import numpy as np
import heapq
from pathlib import Path

from Dataset import Dataset
import NeuMF  # NeuMF.py 안의 get_model 사용


def get_top(userid=1, K=3, dataset_name="study",
            mf_dim=8, layers=(64, 32, 16, 8),
            weight_filename=None):
    """
    NeuMF 가중치로 Top-K 추천 출력
    - Data 폴더: Model/Data
    - Pretrain 폴더: Model/Pretrain
    """

    BASE_DIR = Path(__file__).resolve().parent              # .../Model
    data_prefix = BASE_DIR / "Data" / dataset_name          # .../Model/Data/study
    pretrain_dir = BASE_DIR / "Pretrain"                    # .../Model/Pretrain

    if weight_filename is None:
        weight_filename = f"{dataset_name}_NeuMF.weights.h5"
    weight_path = pretrain_dir / weight_filename

    # 1) 데이터 로드 (Dataset은 prefix + ".train.rating" 등을 내부에서 붙임)
    dataset = Dataset(str(data_prefix))
    train = dataset.trainMatrix
    num_users, num_items = train.shape

    # 2) 모델 생성 (학습과 동일한 구조/파라미터여야 함)
    model = NeuMF.get_model(
        num_users=num_users,
        num_items=num_items,
        mf_dim=mf_dim,
        layers=list(layers),
        reg_layers=[0] * len(layers),
        reg_mf=0
    )

    # 3) 가중치 로드
    if not weight_path.exists():
        raise FileNotFoundError(f"❌ NeuMF weights not found: {weight_path}")

    model.load_weights(str(weight_path))
    print(f"✅ Loaded NeuMF weights: {weight_path}")

    # 4) 전체 item에 대한 예측 점수 (1..num_items-1)
    # (너가 GMF/MLP test에서 하던 방식과 동일하게 0은 제외)
    users = np.array([userid] * (num_items - 1), dtype=np.int32)
    items = np.array(range(1, num_items), dtype=np.int32)

    pred = model.predict([users, items], batch_size=2048, verbose=0).reshape(-1)
    map_item_score = {int(items[i]): float(pred[i]) for i in range(num_items - 1)}
    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    # 5) 사용자 메타 출력(있으면)
    if hasattr(dataset, "usermeta"):
        user_meta_dict = dataset.usermeta
        if userid in user_meta_dict:
            print("User meta:", user_meta_dict[userid])

    print(f"\n[NeuMF] user={userid} Top-{K} 추천:")
    for itemid in ranklist:
        print(f"LP{itemid}", end=" ")
    print()


if __name__ == "__main__":
    # 필요시 userid, K만 바꿔서 사용
    get_top(userid=1, K=3)
