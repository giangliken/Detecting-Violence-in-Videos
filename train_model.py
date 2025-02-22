import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Bước 1: Tải dữ liệu từ các file CSV và gắn nhãn
neutral_data = pd.read_csv("neutral.csv")
kick_data = pd.read_csv("kick.txt")
punch_data = pd.read_csv("punch_0.csv")
choke_data = pd.read_csv("choke_0.csv")
neutral_data['label'] = 'neutral'
kick_data['label'] = 'kick'
punch_data['label'] = 'punch'
choke_data['label'] = 'choke'

# Kết hợp dữ liệu
data = pd.concat([neutral_data, kick_data, punch_data, choke_data], axis=0)

# Chỉ chọn các cột số và loại bỏ cột nhãn
X = data.drop(columns=['label']).select_dtypes(include=['float64', 'int64']).values
y = data['label'].values

# Mã hóa nhãn chuỗi thành số
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Chia dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình với các siêu tham số được điều chỉnh để giảm độ chính xác và độ nhạy
model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=100)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm thử
y_pred = model.predict(X_test)

# Tính các chỉ số đo lường
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
